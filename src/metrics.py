import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .decision import expected_cost


def recall_at_precision(y_true, proba, min_precision: float) -> float:
    """Best recall achievable while keeping precision >= min_precision (0.0 if unattainable)."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def confusion(y_true, flagged) -> dict:
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flagged).astype(bool)
    return {
        "tn": int(((y == 0) & ~f).sum()),
        "fp": int(((y == 0) & f).sum()),
        "fn": int(((y == 1) & ~f).sum()),
        "tp": int(((y == 1) & f).sum()),
    }


def point_metrics(y_true, proba, threshold, cost_fn, cost_fp, min_precision=0.8) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    flagged = p >= threshold
    cm = confusion(y, flagged)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    has_both = 0 < y.sum() < len(y)
    return {
        "pr_auc": float(average_precision_score(y, p)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "recall_at_precision": recall_at_precision(y, p, min_precision) if has_both else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (cm["tn"] + tp) / len(y),
        "expected_cost": expected_cost(y, flagged, cost_fn, cost_fp),
    }


def bootstrap_ci(y_true, proba, threshold, cost_fn, cost_fp, min_precision=0.8,
                 n_boot=1000, ci=0.95, seed=0) -> dict:
    """Percentile bootstrap over transactions. Returns {metric: [lo, hi]}."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    samples = {}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        m = point_metrics(y[idx], p[idx], threshold, cost_fn, cost_fp, min_precision)
        for k, v in m.items():
            samples.setdefault(k, []).append(v)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {k: [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))] for k, v in samples.items()}
