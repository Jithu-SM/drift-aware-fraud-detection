"""
drift_detection.py  — Drift detectors (preserves original structure)
------------------
Adds a third detector (feature drift via KS on key input columns)
so that concept drift in the raw data is caught even before the
model's output distribution shifts.

All public functions keep your original signatures.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

TRANSACTIONS_FILE = "data/transactions.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ Prediction Drift  (your original — unchanged signature)
# ─────────────────────────────────────────────────────────────────────────────

def detect_prediction_drift(window_size: int = 500, threshold: float = 0.05):
    """
    KS test on model output (fraud_probability) between two consecutive windows.

    Returns
    -------
    (drift_detected: bool, info: dict | str)
    """
    df = pd.read_csv(TRANSACTIONS_FILE)

    if len(df) < window_size * 2:
        return False, "Not enough data"

    old = df["fraud_probability"].iloc[-window_size * 2 : -window_size]
    new = df["fraud_probability"].iloc[-window_size:]

    stat, p_value = ks_2samp(old, new)
    drift_detected = bool(stat > threshold)

    return drift_detected, {
        "ks_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
    }


# ──────────────────────────
# 2️⃣ Fraud Rate Drift  
# ──────────────────────────

def detect_fraud_rate_drift(window_size: int = 500, threshold: float = 0.05):
    """
    Compares labelled fraud rate between two consecutive windows.
    Uses admin_label column which reflects any admin corrections.

    Returns
    -------
    (drift_detected: bool, info: dict | str)
    """
    df = pd.read_csv(TRANSACTIONS_FILE)

    if len(df) < window_size * 2:
        return False, "Not enough data"

    old = df["admin_label"].iloc[-window_size * 2 : -window_size]
    new = df["admin_label"].iloc[-window_size:]

    old_rate = (old == "Fraud").mean()
    new_rate = (new == "Fraud").mean()
    change   = abs(new_rate - old_rate)

    drift_detected = bool(change > threshold)

    return drift_detected, {
        "old_fraud_rate": round(old_rate, 4),
        "new_fraud_rate": round(new_rate, 4),
        "change": round(change, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ Feature Drift  (catches concept drift at the raw-data level)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DRIFT_COLS = [
    "velocity_6h",
    "velocity_24h",
    "credit_risk_score",
    "device_fraud_count",
    "intended_balcon_amount",
    "is_night_transaction",
    "foreign_request",
]

def detect_feature_drift(
    window_size: int = 500,
    threshold: float = 0.20,
    min_drifted_features: int = 2,
):
    """
    Runs a KS test on each key feature column between two windows.
    Flags drift when at least `min_drifted_features` columns exceed `threshold`.

    This catches concept drift (change in input patterns) before the model
    output distribution shifts — giving earlier warning.

    Returns
    -------
    (drift_detected: bool, info: dict | str)
    """
    df = pd.read_csv(TRANSACTIONS_FILE)

    if len(df) < window_size * 2:
        return False, "Not enough data"

    available_cols = [c for c in FEATURE_DRIFT_COLS if c in df.columns]
    if not available_cols:
        return False, "No feature columns found in transactions file"

    old = df[available_cols].iloc[-window_size * 2 : -window_size]
    new = df[available_cols].iloc[-window_size:]

    results = {}
    drifted = []
    for col in available_cols:
        stat, p = ks_2samp(old[col].dropna(), new[col].dropna())
        results[col] = {"ks_statistic": round(stat, 4), "p_value": round(p, 4)}
        if stat > threshold:
            drifted.append(col)

    drift_detected = bool(len(drifted) >= min_drifted_features)

    return drift_detected, {
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "details": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined check — used by the API / scheduler
# ─────────────────────────────────────────────────────────────────────────────

def check_all_drift(window_size: int = 500):
    """
    Run all three detectors and return a summary dict.

    Returns
    -------
    {
      "any_drift": bool,
      "prediction": (detected, info),
      "fraud_rate": (detected, info),
      "feature":    (detected, info),
    }
    """
    pred   = detect_prediction_drift(window_size)
    rate   = detect_fraud_rate_drift(window_size)
    feat   = detect_feature_drift(window_size)

    any_drift = bool(pred[0]) or bool(rate[0]) or bool(feat[0])

    def _serializable(result):
        detected, info = result
        return [bool(detected), info if isinstance(info, dict) else {"message": str(info)}]

    return {
        "any_drift":  any_drift,
        "prediction": _serializable(pred),
        "fraud_rate": _serializable(rate),
        "feature":    _serializable(feat),
    }