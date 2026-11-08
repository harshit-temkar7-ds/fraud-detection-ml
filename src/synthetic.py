"""Synthetic stand-in for the Kaggle ULB credit-card dataset.

!!! SYNTHETIC DATA. Metrics measured on it say NOTHING about real-world fraud
performance. It exists so the pipeline, tests and demo run offline. It is
deliberately harder than a naive 'shift two columns' toy: heavy-tailed features,
overlapping classes (25% of frauds look almost normal), night-time fraud bias,
two fraud modes for Amount, and a fraud pattern that weakens over time (so a
time-ordered split is harder than a random split).
"""

import numpy as np
import pandas as pd

from .features import V_COLS

# Std-devs roughly decay like real PCA components.
_SCALES = np.r_[np.linspace(1.9, 1.0, 10), np.linspace(1.0, 0.35, 18)]
# Fraud mean shifts (in units of the feature's std) for informative components.
_SHIFTS = {"V1": -1.0, "V3": -1.2, "V4": 1.1, "V7": -0.7, "V10": -1.3,
           "V11": 1.0, "V12": -1.4, "V14": -1.7, "V16": -0.9, "V17": -1.1}


def generate(n_rows: int = 60000, fraud_rate: float = 0.012, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horizon = 172800.0
    time = np.sort(rng.uniform(0, horizon, n_rows))
    hour = (time % 86400) / 3600.0
    p_fraud = fraud_rate * (1 + 1.2 * np.cos(2 * np.pi * (hour - 3) / 24))
    y = (rng.random(n_rows) < p_fraud).astype(int)

    z = rng.standard_t(df=6, size=(n_rows, 28)) / np.sqrt(6 / 4)  # unit variance
    v = z * _SCALES

    fraud_idx = np.where(y == 1)[0]
    n_fraud = len(fraud_idx)
    hard = rng.random(n_fraud) < 0.25
    strength = np.where(hard, rng.uniform(0.15, 0.4, n_fraud), rng.uniform(0.6, 1.2, n_fraud))
    strength *= 1.0 - 0.25 * (time[fraud_idx] / horizon)  # pattern weakens over time
    for j, col in enumerate(V_COLS):
        shift = _SHIFTS.get(col, 0.0)
        v[fraud_idx, j] = v[fraud_idx, j] * 1.25 + shift * strength * _SCALES[j]

    amount = rng.lognormal(3.4, 1.4, n_rows)
    mode_small = rng.random(n_fraud) < 0.6
    amount[fraud_idx] = np.where(mode_small, rng.lognormal(2.2, 1.2, n_fraud),
                                 rng.lognormal(4.8, 1.0, n_fraud))
    df = pd.DataFrame(np.round(v, 5), columns=V_COLS)
    df.insert(0, "Time", np.round(time, 2))
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y
    return df
