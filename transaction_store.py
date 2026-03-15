"""
transaction_store.py  — Append predictions + handle admin label corrections
--------------------
Keeps data/transactions.csv as the single source of truth for:
  • recent feature values (for drift detection)
  • fraud_probability (model output)
  • model_label    — what the model predicted
  • admin_label    — what the admin says (may differ; used in retraining)
"""

import os
import pandas as pd
import numpy as np
from features import FEATURES

TRANSACTIONS_FILE = "data/transactions.csv"

_COLUMNS = FEATURES + [
    "fraud_probability",
    "model_label",    # "Fraud" | "Legit"
    "admin_label",    # "Fraud" | "Legit"  — starts equal to model_label
    "admin_corrected", # 1 if admin flipped, else 0
]


def _label(prob: float, threshold: float = 0.5) -> str:
    return "Fraud" if prob >= threshold else "Legit"


def append_transaction(features: dict, fraud_prob: float, threshold: float = 0.5) -> None:
    """
    Persist a new prediction.

    Parameters
    ----------
    features    : dict of raw (unscaled, unencoded) feature values
    fraud_prob  : float in [0, 1]
    threshold   : decision boundary
    """
    ml = _label(fraud_prob, threshold)

    row = {**features,
           "fraud_probability": round(float(fraud_prob), 6),
           "model_label":       ml,
           "admin_label":       ml,   # defaults to model prediction
           "admin_corrected":   0}

    new_df = pd.DataFrame([row])

    if os.path.exists(TRANSACTIONS_FILE):
        new_df.to_csv(TRANSACTIONS_FILE, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(TRANSACTIONS_FILE), exist_ok=True)
        new_df.to_csv(TRANSACTIONS_FILE, index=False)


def flip_label(row_index: int) -> dict:
    """
    Admin flips the label for a given row index (0-based).

    Returns the updated row as a dict.
    """
    df = pd.read_csv(TRANSACTIONS_FILE)

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row {row_index} out of range (0..{len(df)-1})")

    current = df.at[row_index, "admin_label"]
    new_lbl = "Legit" if current == "Fraud" else "Fraud"

    df.at[row_index, "admin_label"]    = new_lbl
    df.at[row_index, "admin_corrected"] = 1

    df.to_csv(TRANSACTIONS_FILE, index=False)

    return df.iloc[row_index].to_dict()


def get_recent(n: int = 100) -> pd.DataFrame:
    """Return the n most recent transactions."""
    df = pd.read_csv(TRANSACTIONS_FILE)
    return df.tail(n).reset_index(drop=True)


def count() -> int:
    """Return total number of stored transactions."""
    if not os.path.exists(TRANSACTIONS_FILE):
        return 0
    return sum(1 for _ in open(TRANSACTIONS_FILE)) - 1  # subtract header


def load_all() -> pd.DataFrame:
    if not os.path.exists(TRANSACTIONS_FILE):
        return pd.DataFrame(columns=_COLUMNS)
    return pd.read_csv(TRANSACTIONS_FILE)