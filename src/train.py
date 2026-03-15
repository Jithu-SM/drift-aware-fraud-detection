"""
train.py  — Initial model training (preserves your original structure exactly)
--------------
Reads bank_fraud_dataset.csv, encodes categoricals with LabelEncoder,
scales with StandardScaler, trains XGBClassifier, saves artefacts.

Artefacts written
-----------------
  models/xgb_initial_model.pkl
  models/scaler.pkl
  models/encoders.pkl
"""

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from features import FEATURES

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/bank_fraud_dataset.csv")

# ── Synthetic time features (kept from your original) ────────────────────────
# NOTE: if the CSV already has transaction_hour / is_night_transaction
#       (which generate_datasets.py produces), these lines are no-ops.
if "transaction_hour" not in df.columns:
    df["transaction_hour"] = np.random.randint(0, 24, size=len(df))
if "is_night_transaction" not in df.columns:
    df["is_night_transaction"] = df["transaction_hour"].between(0, 5).astype(int)

# ── Encode categoricals — RAW VALUES ONLY (your comment preserved) ───────────
CATEGORICAL_COLS = ["payment_type", "device_os"]

encoders = {}
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── Feature / target split ───────────────────────────────────────────────────
X = df[FEATURES]
y = df["fraud_bool"]

# ── Scale ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / holdout split ────────────────────────────────────────────────────
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_scaled,
    y,
    test_size=0.2,        # train on 80%
    stratify=y,
    random_state=42
)

# ── Handle class imbalance ───────────────────────────────────────────────────
fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# ── Model ────────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=fraud_ratio,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ── Evaluation on holdout ────────────────────────────────────────────────────
y_pred = model.predict(X_holdout)
print("\nHoldout evaluation")
print(classification_report(y_holdout, y_pred))

# ── Save artefacts ───────────────────────────────────────────────────────────
import os; os.makedirs("models", exist_ok=True)
joblib.dump(model,    "models/xgb_initial_model.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Retraining done with RAW categorical values")