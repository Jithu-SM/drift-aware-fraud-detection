import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

TRANSACTIONS_FILE = "data/transactions.csv"

# -------------------------------
# 1️⃣ Prediction Drift
# -------------------------------
def detect_prediction_drift(window_size=500, threshold=0.15):
    df = pd.read_csv(TRANSACTIONS_FILE)

    if len(df) < window_size * 2:
        return False, "Not enough data"

    old = df["fraud_probability"].iloc[-window_size*2:-window_size]
    new = df["fraud_probability"].iloc[-window_size:]

    stat, p_value = ks_2samp(old, new)

    drift_detected = stat > threshold

    return drift_detected, {
        "ks_statistic": round(stat, 4),
        "p_value": round(p_value, 4)
    }


# -------------------------------
# 2️⃣ Fraud Rate Drift
# -------------------------------
def detect_fraud_rate_drift(window_size=500, threshold=0.05):
    df = pd.read_csv(TRANSACTIONS_FILE)

    if len(df) < window_size * 2:
        return False, "Not enough data"

    old = df["admin_label"].iloc[-window_size*2:-window_size]
    new = df["admin_label"].iloc[-window_size:]

    old_rate = (old == "Fraud").mean()
    new_rate = (new == "Fraud").mean()

    change = abs(new_rate - old_rate)

    drift_detected = change > threshold

    return drift_detected, {
        "old_fraud_rate": round(old_rate, 4),
        "new_fraud_rate": round(new_rate, 4),
        "change": round(change, 4)
    }