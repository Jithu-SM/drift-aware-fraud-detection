import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("data/bank_fraud_dataset.csv")

# Synthetic time features
df["transaction_hour"] = np.random.randint(0, 24, size=len(df))
df["is_night_transaction"] = df["transaction_hour"].between(0, 5).astype(int)

# Feature set
FEATURES = [
    "intended_balcon_amount",
    "payment_type",
    "session_length_in_minutes",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "credit_risk_score",
    "foreign_request",
    "email_is_free",
    "device_fraud_count",
    "device_os",
    "month",
    "transaction_hour",
    "is_night_transaction"
]

TARGET = "fraud_bool"

df = df[FEATURES + [TARGET]]

# Encode categorical fields
encoders = {}
for col in ["payment_type", "device_os"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split: small initial training window
X = df[FEATURES]
y = df[TARGET]

X_train, X_future, y_train, y_future = train_test_split(
    X, y,
    test_size=0.8,
    stratify=y,
    random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluation
train_probs = model.predict_proba(X_train_scaled)[:, 1]
print("ROC-AUC:", roc_auc_score(y_train, train_probs))
print(classification_report(y_train, model.predict(X_train_scaled)))

# Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_initial_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

# Save drift pool
X_future.assign(fraud_bool=y_future).to_csv(
    "data/concept_drift_pool.csv", index=False
)

print("✅ Model trained and drift pool created")
