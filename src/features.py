"""Stateless feature engineering shared by training and serving.

Raw input schema (same as the Kaggle ULB creditcard dataset, minus the label):
    V1..V28  anonymised PCA components
    Amount   transaction amount
    Time     seconds since the first transaction of the dataset
"""

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = V_COLS + ["Amount", "Time"]
DERIVED_FEATURES = ["log_amount", "hour_sin", "hour_cos"]
MODEL_FEATURES = V_COLS + DERIVED_FEATURES
SCALED_FEATURES = ["log_amount"]  # the only column that needs a fitted scaler


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions -> unscaled model features (deterministic, no fitting)."""
    missing = [c for c in RAW_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")
    out = raw[V_COLS].astype(float).copy()
    out["log_amount"] = np.log1p(raw["Amount"].astype(float).clip(lower=0))
    hour = (raw["Time"].astype(float) % 86400) / 3600.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return out[MODEL_FEATURES]
