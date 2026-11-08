import numpy as np
import pytest

from src.decision import expected_cost, optimal_threshold, risk_band_edges, risk_level


def test_threshold_minimises_cost_on_toy_data():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.35, 0.2, 0.1, 0.05, 0.01])
    res = optimal_threshold(y, p, cost_fn=100, cost_fp=1)
    flagged = p >= res["threshold"]
    # cheap false positives: should flag everything >= 0.3 (catches all 3 frauds, 1 FP)
    assert flagged.sum() == 4 and y[flagged].sum() == 3
    assert res["expected_cost"] == expected_cost(y, flagged, 100, 1) == 1


def test_expensive_false_positives_raise_threshold():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.35, 0.2, 0.1, 0.05, 0.01])
    low = optimal_threshold(y, p, cost_fn=100, cost_fp=1)["threshold"]
    high = optimal_threshold(y, p, cost_fn=2, cost_fp=50)["threshold"]
    assert high > low


def test_flag_nothing_when_flagging_never_pays():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.4, 0.5, 0.6, 0.7])  # fraud ranked lowest
    res = optimal_threshold(y, p, cost_fn=1, cost_fp=100)
    assert res["threshold"] == 1.0 and not (p >= res["threshold"]).any()


def test_ties_do_not_split_groups():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    res = optimal_threshold(y, p, cost_fn=10, cost_fp=1)
    assert (p >= res["threshold"]).all()


def test_invalid_input():
    with pytest.raises(ValueError):
        optimal_threshold([], [], 1, 1)


def test_risk_bands_derive_from_threshold():
    t = 0.2
    assert risk_band_edges(t) == {"medium_from": 0.1, "high_from": 0.2}
    assert risk_level(0.05, t) == "LOW"
    assert risk_level(0.1, t) == "MEDIUM"
    assert risk_level(0.199, t) == "MEDIUM"
    assert risk_level(0.2, t) == "HIGH"


def test_committed_threshold_is_used_by_api(client, legit_row, fraud_row):
    from tests.conftest import HEADERS
    info = client.get("/model/info", headers=HEADERS).json()
    r = client.post("/predict", json=fraud_row, headers=HEADERS).json()
    assert r["threshold"] == pytest.approx(info["threshold"], abs=1e-6)
    assert r["is_fraud"] == (r["fraud_probability"] >= info["threshold"])
    assert (r["risk_level"] == "HIGH") == r["is_fraud"]
