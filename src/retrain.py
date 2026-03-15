"""
retrain.py  — Triggered retraining pipeline
-----------
Reads recent transactions (with admin corrections) and retrains
the XGBClassifier using the same hyperparameters as the initial run.

Call trigger_retrain() from app.py whenever drift is detected.
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from features import FEATURES

TRANSACTIONS_FILE = "data/transactions.csv"
CATEGORICAL_COLS  = ["payment_type", "device_os"]

# How many recent rows to use for retraining
RETRAIN_WINDOW = 5_000


def trigger_retrain(verbose: bool = True) -> dict:
    """
    Retrain model on recent transactions.csv data.

    1. Loads RETRAIN_WINDOW most recent rows.
    2. Uses admin_label (which may differ from model_label) as ground truth.
    3. Re-fits encoders + scaler from scratch on this window.
    4. Trains a fresh XGBClassifier and overwrites model artefacts.
    5. Returns evaluation metrics.

    Returns
    -------
    dict with keys: n_samples, fraud_rate, report, retrain_timestamp
    """
    import os, datetime

    df = pd.read_csv(TRANSACTIONS_FILE)

    if len(df) < 200:
        msg = f"Only {len(df)} rows — need at least 200 to retrain."
        print(f"⚠️  {msg}")
        return {"error": msg}

    # Use recent window
    df = df.tail(RETRAIN_WINDOW).copy()

    # Ground truth: admin corrections take precedence
    # admin_label is "Fraud" or "Legit" (string)
    df["label"] = (df["admin_label"] == "Fraud").astype(int)

    if verbose:
        fraud_count = df["label"].sum()
        print(f"🔁 Retraining on {len(df):,} rows | fraud={fraud_count} ({fraud_count/len(df)*100:.1f}%)")

    # ── Encode categoricals ──────────────────────────────────────────────────
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # ── Features & target ────────────────────────────────────────────────────
    available = [f for f in FEATURES if f in df.columns]
    X = df[available]
    y = df["label"]

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Imbalance ────────────────────────────────────────────────────────────
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # ── Model ────────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    report  = classification_report(y_test, y_pred, output_dict=True)
    if verbose:
        print(classification_report(y_test, y_pred))

    # ── Save artefacts (overwrites initial model) ────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,    "models/xgb_initial_model.pkl")
    joblib.dump(scaler,   "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"✅ Retrain complete at {ts}")

    return {
        "n_samples":        len(df),
        "fraud_rate":       round(y.mean(), 4),
        "report":           report,
        "retrain_timestamp": ts,
    }