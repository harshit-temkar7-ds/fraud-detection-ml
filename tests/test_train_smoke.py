import json

import pytest

from src.train import run


def test_tiny_training_outputs(tiny_run):
    model_dir, metrics, _ = tiny_run
    assert (model_dir / "metrics.json").exists() and (model_dir / "reference_stats.json").exists()
    assert metrics["synthetic_data"] is True
    ci = metrics["test"]["pr_auc"]["ci95"]
    assert ci[0] <= metrics["test"]["pr_auc"]["value"] <= ci[1]
    assert set(metrics["cv"]) == {"none", "smote"}
    assert metrics["test_cost_baselines"]["cost_at_chosen_threshold"] <= metrics["test_cost_baselines"]["cost_flag_nothing"] + 1e-6


@pytest.mark.slow
def test_training_reproduces_committed_metrics(tmp_path, committed_model_dir):
    """Re-train with the committed config on the (deterministic) synthetic data; must match metrics.json."""
    from src.config import ROOT, load_config
    from src.synthetic import generate
    committed = json.loads((committed_model_dir / "metrics.json").read_text())
    if not committed.get("synthetic_data"):
        pytest.skip("committed model was trained on real data; this smoke test targets the synthetic demo")
    cfg = load_config(ROOT / "config" / "train.yaml")
    data = tmp_path / "synth.csv"
    generate().to_csv(data, index=False)
    fresh = run(cfg, data, tmp_path / "model", tmp_path / "docs", synthetic=True, update_readme=False)
    assert fresh["selected_strategy"] == committed["selected_strategy"]
    for key in ("pr_auc", "roc_auc", "recall"):
        assert fresh["test"][key]["value"] == pytest.approx(committed["test"][key]["value"], abs=0.02), key
    assert fresh["threshold"] == pytest.approx(committed["threshold"], abs=0.02)


def test_replay_contains_every_test_fraud_in_time_order():
    from src.synthetic import generate
    from src.train import build_replay, time_split
    df = generate(n_rows=20000, fraud_rate=0.002, seed=5).sort_values("Time").reset_index(drop=True)
    _, _, test = time_split(df, 0.6, 0.2)
    replay = build_replay(test, 500, seed=1)
    assert int(replay.Class.sum()) == int(test.Class.sum()) > 0
    assert len(replay) == 500 and replay.Time.is_monotonic_increasing
    # the old 'last N rows' slice would have missed frauds on rare-fraud data
    assert test.tail(500).Class.sum() < test.Class.sum()


def test_replay_weights_reconstruct_population_and_keep_hard_negatives():
    import numpy as np

    from src.synthetic import generate
    from src.train import build_replay, time_split
    df = generate(n_rows=30000, fraud_rate=0.004, seed=11).sort_values("Time").reset_index(drop=True)
    _, _, test = time_split(df, 0.6, 0.2)
    rng = np.random.default_rng(0)
    scores = np.where(test.Class == 1, rng.uniform(0.3, 1, len(test)), rng.beta(0.3, 40, len(test)))
    replay = build_replay(test, 600, seed=2, scores=scores)
    legit = replay[replay.Class == 0]
    assert abs(legit.weight.sum() - (test.Class == 0).sum()) < 1e-3 * len(test)   # weights sum to the legit population
    hard_test = (test.Class == 0) & (scores >= 0.01)
    kept_hard = int((legit.weight == 1.0).sum())
    assert kept_hard == min(int(hard_test.sum()), (600 - int(test.Class.sum())) // 2)
