"""End-to-end smoke check against a RUNNING server. Prints PASS/FAIL per check, exits non-zero on any failure.

    python scripts/smoke_check.py                       # uses http://localhost:8000 and key dev-key-change-me
    python scripts/smoke_check.py --key my-secret --url http://localhost:8000
    python scripts/smoke_check.py --stress              # also tests the rate limiter (sends ~130 requests)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not ok else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default=os.getenv("API_KEY") or (os.getenv("API_KEYS", "").split(",")[0]) or "dev-key-change-me")
    ap.add_argument("--model-dir", default=str(ROOT / "model"))
    ap.add_argument("--stress", action="store_true", help="also test the rate limiter")
    a = ap.parse_args()
    base, H = a.url.rstrip("/"), {"X-API-Key": a.key}
    md = Path(a.model_dir)
    s = requests.Session()

    def get(path, **kw):
        return s.get(base + path, headers=kw.pop("headers", H), timeout=30, **kw)

    def post(path, body=None, **kw):
        return s.post(base + path, json=body, headers=kw.pop("headers", H), timeout=60, **kw)

    print(f"Checking {base}\n")
    try:
        r = get("/live", headers={})
    except requests.RequestException as exc:
        sys.exit(f"Cannot reach {base}: {exc}\nIs the server running? (uvicorn api.main:app --port 8000)")

    # --- probes
    check("/live returns 200", r.status_code == 200)
    r = get("/ready", headers={})
    check("/ready returns 200 (model loaded)", r.status_code == 200, r.text)
    r = get("/health", headers={})
    check("/health reports ready + model_version", r.status_code == 200 and r.json().get("model_loaded") is True, r.text)

    # --- auth
    check("no API key -> 401", post("/predict", {}, headers={}).status_code == 401)
    check("wrong API key -> 401", post("/predict", {}, headers={"X-API-Key": "wrong"}).status_code == 401)
    r = get("/model/info")
    check("correct API key accepted (/model/info 200)", r.status_code == 200, f"{r.status_code} {r.text[:120]}")
    if r.status_code != 200:
        sys.exit("\nAPI key rejected: pass the key the server was started with, e.g. --key dev-key-change-me")
    info = r.json()

    # --- validation
    example = json.loads((ROOT / "example_transaction.json").read_text())
    check("empty body -> 422 (no silent zeros)", post("/predict", {}).status_code == 422)
    check("missing field -> 422", post("/predict", {k: v for k, v in example.items() if k != "V5"}).status_code == 422)
    check("unknown extra field -> 422", post("/predict", {**example, "Class": 1}).status_code == 422)
    check("negative Amount -> 422", post("/predict", {**example, "Amount": -1}).status_code == 422)
    check("out-of-range V3 -> 422", post("/predict", {**example, "V3": 1e9}).status_code == 422)
    check("empty batch -> 422", post("/predict/batch", {"transactions": []}).status_code == 422)
    check("batch of 501 -> 422", post("/predict/batch", {"transactions": [example] * 501}).status_code == 422)

    # --- single prediction
    r = post("/predict", example)
    body = r.json() if r.status_code == 200 else {}
    check("/predict returns 200 with probability in [0,1]", r.status_code == 200 and 0 <= body.get("fraud_probability", -1) <= 1, r.text[:150])
    check("/predict has X-Request-ID header", bool(r.headers.get("X-Request-ID")))
    check("risk level consistent with is_fraud", (body.get("risk_level") == "HIGH") == body.get("is_fraud"))
    check("reasons absent unless explain=true", body.get("reasons") is None)
    r = post("/predict?explain=true", example)
    ok = r.status_code == 200 and (bool(r.json()["reasons"]) if r.json()["is_fraud"] else r.json()["reasons"] is None)
    check("explain=true: reasons present if flagged, absent if not", ok, r.text[:200])

    # --- replay sample: end-to-end check through HTTP vs offline metrics.json
    metrics = json.loads((md / "metrics.json").read_text())
    cm = metrics["test_confusion"]
    replay = pd.read_csv(md / "replay_sample.csv")
    fr = replay[replay.Class == 1]
    legit_hard = replay[(replay.Class == 0) & (replay.get("weight", 1.0) == 1.0)]

    def score(df):
        rows = df[FEATURES].to_dict("records")
        out = []
        for i in range(0, len(rows), 400):
            r = post("/predict/batch?explain=true", {"transactions": rows[i:i + 400]})
            if r.status_code != 200:
                return None
            out += r.json()["results"]
        return out

    res_f = score(fr)
    check("scoring all real test frauds over HTTP works", res_f is not None)
    if res_f is not None:
        tp = sum(x["is_fraud"] for x in res_f)
        check(f"HTTP-scored frauds flagged == metrics.json TP ({cm['tp']})", tp == cm["tp"], f"got {tp}")
        check("every flagged fraud has reason codes", all(bool(x["reasons"]) for x in res_f if x["is_fraud"]))
        check("unflagged rows have no reasons", all(x["reasons"] is None for x in res_f if not x["is_fraud"]))
    res_l = score(legit_hard)
    if res_l is not None:
        fp = sum(x["is_fraud"] for x in res_l)
        check(f"HTTP-scored suspicious legit rows flagged == metrics.json FP ({cm['fp']})", fp == cm["fp"], f"got {fp}")
    else:
        check("scoring suspicious legit rows over HTTP works", False)

    # --- stats / consistency
    st = get("/stats/summary").json()
    check("/stats/summary PR-AUC matches metrics.json", abs(st["pr_auc"] - metrics["test"]["pr_auc"]["value"]) < 1e-9)
    check("/stats/summary confusion matches metrics.json", st.get("confusion") == cm)
    th = get(f"/replay/threshold?value={info['threshold']}").json()
    check("threshold tuner at model threshold reproduces the real confusion matrix",
          (th["tp"], th["fp"], th["fn"]) == (cm["tp"], cm["fp"], cm["fn"]) and th["exact"], json.dumps(th)[:200])
    lo = get("/replay/threshold?value=0.9").json()
    check("higher threshold lowers recall", lo["recall"] < th["recall"])
    check("/replay/threshold rejects value=2 (422)", get("/replay/threshold?value=2").status_code == 422)

    # --- explanations, feed, monitoring
    g = get("/shap/global").json()["feature_importance"]
    check("/shap/global returns 15 ranked features", len(g) == 15 and g[0]["importance"] >= g[-1]["importance"])
    ex = get("/replay/explain?flagged=true").json()
    check("/replay/explain gives 10 TreeSHAP contributions", len(ex.get("contributions", [])) == 10 and ex["is_fraud"])
    feed = get("/replay/feed").json()
    t0 = feed["transactions"][0] if feed.get("transactions") else {}
    check("/replay/feed labelled as replay, no fake merchant/location",
          feed.get("source") == "replay_of_held_out_test_data" and "merchant" not in t0 and "true_label" in t0)
    m = get("/monitoring/metrics").json()
    check("/monitoring/metrics counts predictions and has p95 latency", m["predictions"] > 0 and m["latency_ms"]["p95"] > 0)
    d = get("/monitoring/drift").json()
    check("/monitoring/drift returns a status", d.get("status") in ("insufficient_data", "stable", "moderate_shift", "significant_shift"), str(d)[:120])
    print(f"       drift status: {d.get('status')}  (window samples: {d.get('window_samples')})")

    # --- dashboard + docs
    r = s.get(base + "/", timeout=30)
    check("dashboard page served at /", r.status_code == 200 and "Sentinel" in r.text)
    check("API docs served at /docs", s.get(base + "/docs", timeout=30).status_code == 200)

    # --- error hygiene
    r = post("/predict", {**example, "V1": "NaN-ish"})
    check("422 response does not echo the submitted input", "NaN-ish" not in r.text)

    # --- optional rate limit
    if a.stress:
        codes = [get("/model/info").status_code for _ in range(140)]
        check("rate limiter returns 429 under load", 429 in codes, f"codes seen: {sorted(set(codes))}")
        print("       (wait ~60 s before using the dashboard again)")

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
