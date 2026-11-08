"""Write the synthetic demo dataset (NOT real data) to data/synthetic_creditcard.csv."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.synthetic import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--fraud-rate", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic_creditcard.csv")
    a = ap.parse_args()
    df = generate(a.rows, a.fraud_rate, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"SYNTHETIC data written to {a.out}: {len(df):,} rows, {int(df.Class.sum()):,} fraud")
