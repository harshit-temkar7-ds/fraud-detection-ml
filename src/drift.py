"""Population Stability Index (PSI) helpers."""

import numpy as np

EPS = 1e-4


def reference_bins(values, n_bins: int = 10) -> dict:
    """Quantile bin edges + reference proportions from training data."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    inner = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.r_[-np.inf, inner, np.inf]
    counts, _ = np.histogram(v, bins=edges)
    props = counts / counts.sum()
    return {"edges": [None if np.isinf(e) else float(e) for e in edges], "proportions": props.tolist()}


def _edges(ref: dict) -> np.ndarray:
    e = ref["edges"]
    last = len(e) - 1
    return np.array([-np.inf if i == 0 else np.inf if i == last else x for i, x in enumerate(e)], dtype=float)


def psi(ref: dict, new_values) -> float:
    v = np.asarray(new_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to compare")
    counts, _ = np.histogram(v, bins=_edges(ref))
    actual = np.clip(counts / counts.sum(), EPS, None)
    expected = np.clip(np.asarray(ref["proportions"], dtype=float), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate_shift"
    return "significant_shift"
