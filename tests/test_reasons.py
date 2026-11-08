import numpy as np

from src.reasons import build_reasons

NAMES = ["V1", "V2", "log_amount", "hour_sin", "hour_cos"]
STATS = {n: {"mean": 0.0, "std": 1.0} for n in NAMES}


def test_only_positive_contributions_become_reasons():
    reasons = build_reasons(NAMES, np.array([0.9, -2.0, 0.0, 0.0, 0.0]), np.array([-3.0, 1.0, 0, 0, 0]), STATS)
    assert [r["feature"] for r in reasons] == ["V1"]
    assert reasons[0]["code"] == "R_V1_LOW"


def test_time_features_are_merged():
    reasons = build_reasons(NAMES, np.array([0, 0, 0, 0.4, 0.5]), np.zeros(5), STATS, hour=3)
    assert len(reasons) == 1 and reasons[0]["code"] == "R_TIME_OF_DAY"
    assert abs(reasons[0]["contribution"] - 0.9) < 1e-6 and "03:00" in reasons[0]["description"]


def test_amount_direction_and_top_k():
    reasons = build_reasons(NAMES, np.array([0.1, 0.2, 0.9, 0.3, 0]), np.array([0, 0, 2.5, 0, 0]), STATS, top_k=2)
    assert reasons[0]["code"] == "R_AMOUNT_HIGH" and len(reasons) == 2
