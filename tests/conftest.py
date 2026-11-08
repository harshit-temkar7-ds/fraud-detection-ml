import copy
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402
from src.config import load_config  # noqa: E402
from src.features import RAW_FEATURES  # noqa: E402
from src.synthetic import generate  # noqa: E402

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture(scope="session")
def committed_model_dir():
    d = ROOT / "model"
    if not (d / "metadata.json").exists():
        pytest.skip("no committed model artifacts; run `python -m src.train --synthetic`")
    return d


@pytest.fixture(scope="session")
def replay_df(committed_model_dir):
    return pd.read_csv(committed_model_dir / "replay_sample.csv")


@pytest.fixture()
def make_client(committed_model_dir):
    clients = []

    def _make(model_dir=None, **overrides):
        base = dict(model_dir=model_dir or committed_model_dir, api_keys=(KEY,), auth_disabled=False,
                    rate_limit_per_min=10_000, drift_min_samples=20)
        base.update(overrides)
        client = TestClient(create_app(Settings(**base)), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(make_client):
    return make_client()


@pytest.fixture()
def fraud_row(replay_df):
    return replay_df[replay_df.Class == 1].iloc[0][RAW_FEATURES].to_dict()


@pytest.fixture()
def legit_row(replay_df):
    return replay_df[replay_df.Class == 0].iloc[0][RAW_FEATURES].to_dict()


@pytest.fixture(scope="session")
def tiny_cfg():
    cfg = load_config(ROOT / "config" / "train.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["xgb"].update(n_estimators=25, n_jobs=1)
    cfg["cv"]["folds"] = 2
    cfg["strategies"] = ["none", "smote"]
    cfg["bootstrap"]["n"] = 30
    cfg["replay_rows"] = 200
    return cfg


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, tiny_cfg):
    """Train a tiny model once per session on 6k synthetic rows. Returns (model_dir, metrics, data_path)."""
    from src.train import run
    root = tmp_path_factory.mktemp("tiny")
    data_path = root / "tiny.csv"
    generate(n_rows=6000, fraud_rate=0.03, seed=3).to_csv(data_path, index=False)
    metrics = run(tiny_cfg, data_path, root / "model", root / "docs", synthetic=True, update_readme=False)
    return root / "model", metrics, data_path
