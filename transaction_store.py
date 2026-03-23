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
    Admin flips the label for a given ABSOLUTE CSV row index.
    Always use the csv_row field returned by get_recent(), not
    the position in the fetched slice.

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
    """Return the n most recent transactions with their real CSV row index.
    Always guarantees model_label, admin_label, admin_corrected, and
    fraud_probability columns exist — even if the CSV was written by an
    older version of the code that lacked them.
    """
    df = pd.read_csv(TRANSACTIONS_FILE)

    df    = _ensure_columns(df)
    tail  = df.tail(n).copy()
    tail["csv_row"] = tail.index   # actual row number in the CSV file
    return tail.reset_index(drop=True)


def count() -> int:
    """Return total number of stored transactions."""
    if not os.path.exists(TRANSACTIONS_FILE):
        return 0
    return sum(1 for _ in open(TRANSACTIONS_FILE)) - 1  # subtract header


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee required columns exist with safe defaults (handles old CSV files)."""
    if "fraud_probability" not in df.columns:
        df["fraud_probability"] = 0.0
    if "model_label" not in df.columns:
        df["model_label"] = df["fraud_probability"].apply(
            lambda p: "Fraud" if float(p) >= 0.5 else "Legit"
        )
    if "admin_label" not in df.columns:
        df["admin_label"] = df["model_label"]
    if "admin_corrected" not in df.columns:
        df["admin_corrected"] = 0
    df["fraud_probability"]  = pd.to_numeric(df["fraud_probability"],  errors="coerce").fillna(0.0)
    df["model_label"]        = df["model_label"].fillna("Legit").astype(str)
    df["admin_label"]        = df["admin_label"].fillna(df["model_label"]).astype(str)
    df["admin_corrected"]    = pd.to_numeric(df["admin_corrected"], errors="coerce").fillna(0).astype(int)
    return df


def load_all() -> pd.DataFrame:
    if not os.path.exists(TRANSACTIONS_FILE):
        return pd.DataFrame(columns=_COLUMNS)
    return _ensure_columns(pd.read_csv(TRANSACTIONS_FILE))