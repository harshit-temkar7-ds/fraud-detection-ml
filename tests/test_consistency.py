import json
import shutil

import numpy as np
import pytest

from src.features import MODEL_FEATURES, RAW_FEATURES, make_features
from src.model_bundle import ArtifactError, ModelBundle


def test_features_are_deterministic_and_ordered(replay_df):
    a = make_features(replay_df[RAW_FEATURES])
    b = make_features(replay_df[RAW_FEATURES].sample(frac=1, random_state=0).sort_index())
    assert list(a.columns) == MODEL_FEATURES
    assert np.allclose(a.values, b.values)


def test_committed_bundle_loads_and_orders_features(committed_model_dir, replay_df):
    bundle = ModelBundle.load(committed_model_dir)
    assert sorted(bundle.feature_names) == sorted(MODEL_FEATURES)
    prepared = bundle.prepare(replay_df[RAW_FEATURES].head(5))
    assert list(prepared.columns) == bundle.feature_names
    p = bundle.predict_proba(replay_df[RAW_FEATURES].head(5))
    assert ((p >= 0) & (p <= 1)).all()


def test_scaling_uses_train_statistics(committed_model_dir, replay_df):
    bundle = ModelBundle.load(committed_model_dir)
    prepared = bundle.prepare(replay_df[RAW_FEATURES])
    expected = (np.log1p(replay_df["Amount"]) - bundle.scaler_mean[0]) / bundle.scaler_scale[0]
    assert np.allclose(prepared["log_amount"], expected)


def test_saved_bundle_matches_training_pipeline(tiny_run):
    """train.run() itself raises if bundle != pipeline; assert artifacts + checksums here."""
    model_dir, metrics, _ = tiny_run
    bundle = ModelBundle.load(model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert set(meta["artifact_sha256"]) == {"xgb_model.json", "preprocess.json", "calibrator.json"}
    assert bundle.threshold == pytest.approx(metrics["threshold"])
    assert not any(p.suffix == ".pkl" for p in model_dir.iterdir()), "no pickles"


def test_contributions_sum_to_margin(committed_model_dir, replay_df):
    import xgboost as xgb
    bundle = ModelBundle.load(committed_model_dir)
    raw = replay_df[RAW_FEATURES].head(10)
    prepared, contrib = bundle.contributions(raw)
    booster = bundle.clf.get_booster()
    full = booster.predict(xgb.DMatrix(prepared, feature_names=bundle.feature_names), pred_contribs=True)
    margin = booster.predict(xgb.DMatrix(prepared, feature_names=bundle.feature_names), output_margin=True)
    assert np.allclose(full.sum(axis=1), margin, atol=1e-4)


def test_tampered_artifact_is_rejected(tiny_run, tmp_path):
    model_dir, _, _ = tiny_run
    copy = tmp_path / "m"
    shutil.copytree(model_dir, copy)
    (copy / "calibrator.json").write_text('{"method": "none"}')
    with pytest.raises(ArtifactError, match="Checksum"):
        ModelBundle.load(copy)


def test_missing_artifacts_rejected(tmp_path):
    with pytest.raises(ArtifactError):
        ModelBundle.load(tmp_path)


def test_strict_version_mismatch(tiny_run, tmp_path):
    from src.config import sha256_file  # noqa: F401
    model_dir, _, _ = tiny_run
    copy = tmp_path / "m"
    shutil.copytree(model_dir, copy)
    meta = json.loads((copy / "metadata.json").read_text())
    meta["library_versions"]["xgboost"] = "0.1.0"
    (copy / "metadata.json").write_text(json.dumps(meta))
    ModelBundle.load(copy)  # tolerated (warning)
    with pytest.raises(ArtifactError, match="xgboost version mismatch"):
        ModelBundle.load(copy, strict_versions=True)


def test_time_split_is_ordered_and_disjoint(tiny_run):
    import pandas as pd

    from src.train import load_dataset, time_split
    _, _, data_path = tiny_run
    tr, va, te = time_split(load_dataset(data_path), 0.6, 0.2)
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()
    assert len(pd.concat([tr, va, te])) == 6000
