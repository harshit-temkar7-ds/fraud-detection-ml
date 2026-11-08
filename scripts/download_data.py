"""Download the real Kaggle ULB credit-card fraud dataset into data/creditcard.csv.

Requires a Kaggle account + API token (https://www.kaggle.com/settings -> 'Create New Token'),
saved as ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY env vars).

    pip install kaggle
    python scripts/download_data.py

Manual alternative: download https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud,
unzip, and place creditcard.csv in data/. Then run:  python scripts/download_data.py --check-only
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

EXPECTED_ROWS, EXPECTED_FRAUD = 284807, 492
DEST = Path("data/creditcard.csv")


def validate(path: Path) -> None:
    df = pd.read_csv(path, usecols=["Class"])
    rows, fraud = len(df), int(df["Class"].sum())
    if (rows, fraud) != (EXPECTED_ROWS, EXPECTED_FRAUD):
        sys.exit(f"Unexpected file: {rows:,} rows / {fraud} fraud "
                 f"(expected {EXPECTED_ROWS:,} / {EXPECTED_FRAUD}). Is this the ULB dataset?")
    print(f"OK: {path} has {rows:,} rows and {fraud} fraud cases.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if not args.check_only:
        if shutil.which("kaggle") is None:
            sys.exit("The 'kaggle' CLI is not installed. Run: pip install kaggle  (or download manually; see docstring)")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud",
                        "-p", str(DEST.parent), "--unzip"], check=True)
    if not DEST.exists():
        sys.exit(f"{DEST} not found.")
    validate(DEST)
