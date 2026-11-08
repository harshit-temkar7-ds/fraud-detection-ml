import numpy as np

from src.drift import psi, psi_status, reference_bins


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = reference_bins(rng.normal(size=20000))
    assert psi(ref, rng.normal(size=5000)) < 0.05


def test_psi_large_for_shift():
    rng = np.random.default_rng(0)
    ref = reference_bins(rng.normal(size=20000))
    value = psi(ref, rng.normal(loc=2.0, size=5000))
    assert value > 0.25 and psi_status(value) == "significant_shift"


def test_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "moderate_shift"
    assert psi_status(0.3) == "significant_shift"


def test_reference_serialisable_roundtrip():
    import json
    ref = json.loads(json.dumps(reference_bins(np.arange(1000.0))))
    assert psi(ref, np.arange(1000.0)) < 1e-6
