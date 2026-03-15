import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from features import FEATURES

df = pd.read_csv("data/bank_fraud_dataset.csv")

# Synthetic time features
df["transaction_hour"] = np.random.randint(0, 24, size=len(df))
df["is_night_transaction"] = df["transaction_hour"].between(0, 5).astype(int)

# IMPORTANT: NO MAPPING, RAW VALUES ONLY
CATEGORICAL_COLS = ["payment_type", "device_os"]

encoders = {}

for col in CATEGORICAL_COLS:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[FEATURES]
y = df["fraud_bool"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_scaled,
    y,
    test_size=0.2,   # train on 80%
    stratify=y,
    random_state=42
)

fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=fraud_ratio,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


joblib.dump(model, "models/xgb_initial_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Retraining done with RAW categorical values")
