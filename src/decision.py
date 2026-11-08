"""Cost-based decision threshold and risk bands."""

import numpy as np


def optimal_threshold(y_true, proba, cost_fn: float, cost_fp: float) -> dict:
    """Pick the threshold that minimises  FN*cost_fn + FP*cost_fp.

    A transaction is flagged when proba >= threshold. Candidate thresholds are
    the boundaries between distinct score values (midpoints), so the chosen
    threshold is not glued to a single validation score.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and proba must be non-empty and the same length")

    n_pos = int(y.sum())
    baseline = n_pos * cost_fn  # cost of flagging nothing
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last_of_group = np.where(np.r_[ps[1:] != ps[:-1], True])[0]
    cost = (n_pos - tp[last_of_group]) * cost_fn + fp[last_of_group] * cost_fp

    best = int(np.argmin(cost))
    if cost[best] >= baseline:
        # Flagging anything costs more than it saves: flag nothing.
        return {"threshold": 1.0, "expected_cost": float(baseline), "baseline_cost": float(baseline)}

    idx = last_of_group[best]
    if idx + 1 < len(ps):
        threshold = float((ps[idx] + ps[idx + 1]) / 2.0)
    else:
        threshold = float(ps[idx] / 2.0)
    return {"threshold": threshold, "expected_cost": float(cost[best]), "baseline_cost": float(baseline)}


def expected_cost(y_true, flagged, cost_fn: float, cost_fp: float) -> float:
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flagged).astype(bool)
    fn = int(((y == 1) & ~f).sum())
    fp = int(((y == 0) & f).sum())
    return float(fn * cost_fn + fp * cost_fp)


def risk_band_edges(threshold: float) -> dict:
    """Risk bands derive from the decision threshold t:
    LOW < t/2 <= MEDIUM < t <= HIGH.   HIGH is exactly the 'flagged as fraud' set.
    """
    return {"medium_from": threshold * 0.5, "high_from": threshold}


def risk_level(probability: float, threshold: float) -> str:
    edges = risk_band_edges(threshold)
    if probability >= edges["high_from"]:
        return "HIGH"
    if probability >= edges["medium_from"]:
        return "MEDIUM"
    return "LOW"
