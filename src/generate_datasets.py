"""
generate_datasets.py
--------------------
Generates two CSVs that match the FEATURES list in features.py:

  data/bank_fraud_dataset.csv  — normal distribution, ~5% fraud
  data/drift_dataset.csv       — shifted distribution designed to trigger
                                  both prediction drift and fraud-rate drift
"""

import numpy as np
import pandas as pd
from constants import PAYMENT_TYPES, DEVICE_OS

RNG = np.random.default_rng(42)


def _base_frame(n: int, fraud_rate: float, rng) -> pd.DataFrame:
    is_fraud = (rng.random(n) < fraud_rate).astype(int)

    # --- session / velocity features ---
    session_length = rng.exponential(scale=8, size=n).clip(0.5, 60)
    velocity_6h    = rng.poisson(lam=2, size=n).clip(0, 20).astype(float)
    velocity_24h   = (velocity_6h * rng.uniform(1, 4, size=n)).clip(0, 80)
    velocity_4w    = (velocity_24h * rng.uniform(3, 10, size=n)).clip(0, 500)

    # fraud amplifies velocity
    velocity_6h  += is_fraud * rng.integers(3, 12, size=n)
    velocity_24h += is_fraud * rng.integers(5, 30, size=n)

    # --- balance / amount features ---
    intended_balcon_amount = rng.exponential(scale=300, size=n) * (1 + 2 * is_fraud)

    # --- risk / device features ---
    credit_risk_score = rng.integers(100, 900, size=n) - is_fraud * rng.integers(50, 200, size=n)
    credit_risk_score = credit_risk_score.clip(50, 999)

    foreign_request    = (rng.random(n) < 0.05 + 0.25 * is_fraud).astype(int)
    email_is_free      = (rng.random(n) < 0.4  + 0.35 * is_fraud).astype(int)
    device_fraud_count = rng.poisson(lam=0.1, size=n) + is_fraud * rng.integers(0, 5, size=n)

    # --- categorical ---
    payment_type = rng.choice(PAYMENT_TYPES, size=n,
                              p=[0.40, 0.25, 0.20, 0.10, 0.05])
    device_os    = rng.choice(DEVICE_OS, size=n,
                              p=[0.30, 0.35, 0.20, 0.10, 0.05])

    # --- time features ---
    month               = rng.integers(1, 13, size=n)
    transaction_hour    = rng.integers(0, 24, size=n)
    is_night_transaction = ((transaction_hour >= 0) & (transaction_hour <= 5)).astype(int)

    return pd.DataFrame({
        "intended_balcon_amount": intended_balcon_amount.round(2),
        "payment_type":           payment_type,
        "session_length_in_minutes": session_length.round(2),
        "velocity_6h":            velocity_6h,
        "velocity_24h":           velocity_24h.round(1),
        "velocity_4w":            velocity_4w.round(1),
        "credit_risk_score":      credit_risk_score,
        "foreign_request":        foreign_request,
        "email_is_free":          email_is_free,
        "device_fraud_count":     device_fraud_count,
        "device_os":              device_os,
        "month":                  month,
        "transaction_hour":       transaction_hour,
        "is_night_transaction":   is_night_transaction,
        "fraud_bool":             is_fraud,
    })


def generate_normal(n=10_000, path="data/bank_fraud_dataset.csv"):
    """Baseline training dataset — ~5 % fraud, normal pattern."""
    df = _base_frame(n, fraud_rate=0.05, rng=RNG)
    df.to_csv(path, index=False)
    fraud_count = df["fraud_bool"].sum()
    print(f"[normal]  {n:,} rows | fraud={fraud_count} ({fraud_count/n*100:.1f}%) → {path}")
    return df


def generate_drift(n=3_000, path="data/drift_dataset.csv"):
    """
    Drift dataset — designed to trigger BOTH detectors:

    Concept drift levers
    --------------------
    1. Fraud rate jumps from 5% → 30% (fraud_rate_drift threshold=0.05)
    2. Prediction score distribution shifts upward (KS threshold=0.15)
       - achieved by: high velocity, high device_fraud_count, night txns, foreign reqs
    3. Payment-type distribution flips: CASH_OUT dominates (unusual pattern)
    4. Most transactions happen 00-04 h (night shift)
    """
    rng_drift = np.random.default_rng(99)

    df = _base_frame(n, fraud_rate=0.30, rng=rng_drift)

    # --- shift patterns that the model hasn't seen ---
    # 1. CASH_OUT dominates (was 10% → now 65%)
    df["payment_type"] = rng_drift.choice(
        PAYMENT_TYPES, size=n, p=[0.05, 0.05, 0.15, 0.65, 0.10]
    )
    # 2. Mostly night transactions
    df["transaction_hour"]    = rng_drift.integers(0, 6, size=n)
    df["is_night_transaction"] = 1

    # 3. Spike velocity & device fraud count for everyone
    df["velocity_6h"]         = (df["velocity_6h"]  * rng_drift.uniform(2, 5, size=n)).clip(0, 100)
    df["velocity_24h"]        = (df["velocity_24h"] * rng_drift.uniform(2, 4, size=n)).clip(0, 300)
    df["device_fraud_count"]  = df["device_fraud_count"] + rng_drift.integers(2, 8, size=n)

    # 4. Credit scores universally lower
    df["credit_risk_score"]   = (df["credit_risk_score"] * 0.6).clip(50, 999).astype(int)

    # 5. Foreign requests much more common
    df["foreign_request"]     = (rng_drift.random(n) < 0.45).astype(int)

    df.to_csv(path, index=False)
    fraud_count = df["fraud_bool"].sum()
    print(f"[drift]   {n:,} rows | fraud={fraud_count} ({fraud_count/n*100:.1f}%) → {path}")
    return df


if __name__ == "__main__":
    generate_normal()
    generate_drift()
    print("\nDatasets ready. Run train.py next.")