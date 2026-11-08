"""Build a Hugging Face *model repo* folder (default: hf_model/) from the trained artifacts.

Contains the pickle-free model files, metrics and the generated model card (as README.md with HF metadata).
The replay sample (rows of the Kaggle dataset) is intentionally NOT included.

    python scripts/make_hf_model_repo.py
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ("xgb_model.json", "preprocess.json", "calibrator.json", "metadata.json", "metrics.json", "reference_stats.json")

FRONT = """---
license: mit
library_name: xgboost
pipeline_tag: tabular-classification
tags:
  - fraud-detection
  - tabular-classification
  - xgboost
  - calibrated-probabilities
  - treeshap
---

"""

HOWTO = """

## How to use these files

The model is stored without pickle: `xgb_model.json` (XGBoost native), `preprocess.json` (scaler parameters),
`calibrator.json` (probability calibrator) and `metadata.json` (feature order, decision threshold, costs, checksums).
Load them with `ModelBundle.load()` from the Sentinel source code (`src/model_bundle.py` in the GitHub repository).

Data attribution: Kaggle *Credit Card Fraud Detection* (ULB / Worldline), DbCL v1.0 — see `NOTICE.md`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "hf_model"))
    a = ap.parse_args()
    src, out = ROOT / "model", Path(a.out)
    card = ROOT / "docs" / "MODEL_CARD.md"
    missing = [f for f in FILES if not (src / f).exists()]
    if missing or not card.exists():
        sys.exit(f"Missing {missing or 'docs/MODEL_CARD.md'}. Run: python -m src.train")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in FILES:
        shutil.copy(src / f, out / f)
    if (ROOT / "docs" / "figures").exists():
        shutil.copytree(ROOT / "docs" / "figures", out / "figures")
    if (ROOT / "NOTICE.md").exists():
        shutil.copy(ROOT / "NOTICE.md", out / "NOTICE.md")
    text = card.read_text(encoding="utf-8")
    (out / "README.md").write_text(FRONT + text + HOWTO, encoding="utf-8")
    print(f"Model repo folder ready: {out}")


if __name__ == "__main__":
    main()
