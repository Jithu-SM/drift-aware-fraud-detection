"""
predict.py  — Inference helpers (preserves your original structure)
-----------
Loads the current model from disk. Call reload_model() after retraining
to hot-swap without restarting the process.
"""

import joblib
import numpy as np

_model   = None
_scaler  = None
_encoders = None


def _load():
    global _model, _scaler, _encoders
    _model    = joblib.load("models/xgb_initial_model.pkl")
    _scaler   = joblib.load("models/scaler.pkl")
    _encoders = joblib.load("models/encoders.pkl")


def reload_model():
    """Hot-swap the in-memory model after retraining."""
    _load()
    print("🔄 Model hot-swapped from disk.")


# Load on import
_load()


def predict_fraud(X_scaled: np.ndarray) -> np.ndarray:
    """Return fraud probability for each row in X_scaled (already encoded + scaled)."""
    return _model.predict_proba(X_scaled)[:, 1]


def preprocess(raw: dict) -> np.ndarray:
    """
    Convert a raw transaction dict → scaled numpy array ready for predict_fraud().

    raw must contain all keys in FEATURES with pre-encoded numeric values
    OR the string categoricals (payment_type, device_os) which will be
    label-encoded here using the saved encoders.
    """
    import pandas as pd
    from features import FEATURES

    df = pd.DataFrame([raw])

    for col, le in _encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    X = df[FEATURES].values
    return _scaler.transform(X)