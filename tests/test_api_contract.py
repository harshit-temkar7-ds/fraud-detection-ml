import pytest

from tests.conftest import HEADERS


def test_predict_contract(client, fraud_row):
    r = client.post("/predict", json=fraud_row, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"transaction_id", "is_fraud", "fraud_probability", "risk_level", "threshold",
                         "amount", "timestamp", "model_version", "reasons"}
    assert 0 <= body["fraud_probability"] <= 1
    assert body["reasons"] is None            # explain not requested
    assert r.headers["x-request-id"]


def test_explain_only_for_flagged(client, fraud_row, legit_row):
    flagged = client.post("/predict?explain=true", json=fraud_row, headers=HEADERS).json()
    assert flagged["is_fraud"] and flagged["reasons"]
    assert flagged["reasons"][0]["code"].startswith("R_")
    legit = client.post("/predict?explain=true", json=legit_row, headers=HEADERS).json()
    assert not legit["is_fraud"] and legit["reasons"] is None


def test_batch_contract_and_explain(client, fraud_row, legit_row):
    body = {"transactions": [legit_row, fraud_row, legit_row]}
    r = client.post("/predict/batch?explain=true", json=body, headers=HEADERS).json()
    assert r["total"] == 3 and len(r["results"]) == 3
    assert r["fraud_count"] == sum(x["is_fraud"] for x in r["results"])
    assert r["reasons_note"]
    assert all((x["reasons"] is not None) == x["is_fraud"] for x in r["results"])


def test_request_id_is_echoed_and_sanitised(client):
    r = client.get("/live", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = client.get("/live", headers={"X-Request-ID": "bad id with spaces!"})
    assert r.headers["x-request-id"] != "bad id with spaces!"


def test_model_missing_gives_503_not_random_scores(make_client, legit_row, tmp_path):
    c = make_client(model_dir=tmp_path)
    assert c.get("/live").status_code == 200
    assert c.get("/ready").status_code == 503
    h = c.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "not_ready"
    assert c.post("/predict", json=legit_row, headers=HEADERS).status_code == 503
    assert c.get("/replay/feed", headers=HEADERS).status_code == 503


def test_unhandled_error_is_generic(make_client, legit_row):
    c = make_client()
    state = c.app.state.sentinel

    def boom(_):
        raise RuntimeError("secret internal path /etc/shadow")

    state.bundle.predict_proba = boom
    r = c.post("/predict", json=legit_row, headers=HEADERS)
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["detail"] == "Internal server error"
    assert r.json()["request_id"]


def test_replay_feed_is_labelled_replay(client):
    r = client.get("/replay/feed", headers=HEADERS).json()
    assert r["source"] == "replay_of_held_out_test_data" and r["count"] == 20
    row = r["transactions"][0]
    assert "merchant" not in row and "location" not in row and "true_label" in row


def test_stats_and_shap_and_info(client):
    s = client.get("/stats/summary", headers=HEADERS).json()
    assert s["pr_auc"] > 0 and "confusion" in s
    g = client.get("/shap/global", headers=HEADERS).json()["feature_importance"]
    assert len(g) == 15 and g[0]["importance"] >= g[-1]["importance"]
    assert client.get("/model/info", headers=HEADERS).json()["features"] == 31


def test_monitoring_and_drift(client, replay_df, legit_row):
    from src.features import RAW_FEATURES
    assert client.get("/monitoring/drift", headers=HEADERS).json()["status"] == "insufficient_data"
    rows = replay_df.head(200)[RAW_FEATURES].to_dict("records")
    assert client.post("/predict/batch", json={"transactions": rows}, headers=HEADERS).status_code == 200
    m = client.get("/monitoring/metrics", headers=HEADERS).json()
    assert m["predictions"] == 200 and m["latency_ms"]["p95"] >= m["latency_ms"]["p50"] > 0
    d = client.get("/monitoring/drift", headers=HEADERS).json()
    assert d["status"] in ("stable", "moderate_shift", "significant_shift")
    assert "Amount" in d["features"]
    # push clearly shifted amounts through -> drift must be detected
    shifted = [{**legit_row, "Amount": 5000.0} for _ in range(300)]
    client.post("/predict/batch", json={"transactions": shifted[:300]}, headers=HEADERS)
    d2 = client.get("/monitoring/drift", headers=HEADERS).json()
    assert d2["features"]["Amount"]["psi"] > d["features"]["Amount"]["psi"]


def test_dashboard_served(client):
    assert client.get("/").status_code == 200


def test_replay_threshold_is_measured_not_invented(client):
    lo = client.get("/replay/threshold?value=0.01", headers=HEADERS).json()
    hi = client.get("/replay/threshold?value=0.9", headers=HEADERS).json()
    assert lo["tp"] + lo["fn"] == lo["frauds"] == hi["tp"] + hi["fn"]
    assert lo["recall"] >= hi["recall"] and lo["fp"] >= hi["fp"]
    assert client.get("/replay/threshold?value=2", headers=HEADERS).status_code == 422


def test_replay_explain_returns_real_contributions(client):
    r = client.get("/replay/explain?flagged=true", headers=HEADERS).json()
    assert r["is_fraud"] and len(r["contributions"]) == 10 and r["reasons"]
    ok = client.get("/replay/explain?flagged=false", headers=HEADERS).json()
    assert not ok["is_fraud"]


def test_replay_threshold_reconstructs_the_real_test_split(client, committed_model_dir):
    """At thresholds >= the lowest 'hard' legit score, the replay reproduces the REAL test confusion matrix exactly."""
    import json

    from src.features import RAW_FEATURES
    from src.model_bundle import ModelBundle
    from src.synthetic import generate
    from src.train import time_split

    meta = json.loads((committed_model_dir / "metadata.json").read_text())
    if not meta["data"]["synthetic"]:
        pytest.skip("committed model not trained on the synthetic set; cannot recompute the truth offline")
    df = generate().sort_values("Time", kind="stable").reset_index(drop=True)
    _, _, test = time_split(df, 0.6, 0.2)
    proba = ModelBundle.load(committed_model_dir).predict_proba(test[RAW_FEATURES])
    y = test["Class"].values

    t = meta["threshold"]
    first = client.get(f"/replay/threshold?value={t}", headers=HEADERS).json()
    assert first["weighted"] and first["exact"]
    for value in (t, 0.1, 0.3, 0.6, 0.9):
        r = client.get(f"/replay/threshold?value={value}", headers=HEADERS).json()
        flagged = proba >= value
        truth = {"tp": int((flagged & (y == 1)).sum()), "fp": int((flagged & (y == 0)).sum()),
                 "fn": int((~flagged & (y == 1)).sum())}
        assert r["exact"], value
        assert (r["tp"], r["fp"], r["fn"]) == (truth["tp"], truth["fp"], truth["fn"]), value
        prec = truth["tp"] / max(truth["tp"] + truth["fp"], 1)
        assert r["precision"] == pytest.approx(prec, abs=1e-3)
    # below the exact range the answer is still sane but flagged as an estimate
    low = client.get("/replay/threshold?value=0.0", headers=HEADERS).json()
    assert low["fp"] > first["fp"]
