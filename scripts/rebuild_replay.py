"""Rebuild model/replay_sample.csv (all test frauds + all model-suspicious legit rows + weighted easy rows)
WITHOUT re-training. Needs the saved model in model/ to score the test split.

    python scripts/rebuild_replay.py            # uses config/train.yaml and data/creditcard.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, load_config  # noqa: E402
from src.features import RAW_FEATURES  # noqa: E402
from src.model_bundle import ModelBundle  # noqa: E402
from src.train import build_replay, load_dataset, time_split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "train.yaml"))
    ap.add_argument("--data")
    ap.add_argument("--model-dir", default=str(ROOT / "model"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = load_dataset(Path(a.data or ROOT / cfg["data"]["path"]))
    _, _, test = time_split(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"])
    bundle = ModelBundle.load(a.model_dir)
    scores = bundle.predict_proba(test[RAW_FEATURES])
    replay = build_replay(test, int(cfg.get("replay_rows", 3000)), cfg["seed"], scores=scores)
    Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    out = Path(a.model_dir) / "replay_sample.csv"
    replay.to_csv(out, index=False)
    hard = int(((replay.Class == 0) & (replay.weight == 1.0)).sum())
    print(f"Wrote {out}: {len(replay):,} rows, {int(replay.Class.sum())} frauds, {hard} exact (weight 1) legit rows")
