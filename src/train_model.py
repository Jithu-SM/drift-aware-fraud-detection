import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(
    "data/fraud.csv",
    usecols=[
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud"
    ]
)

# -------------------------------
# Feature Engineering
# -------------------------------

# 1️⃣ Convert step to transaction hour
# Each step = 1 hour
df["transaction_hour"] = df["step"] % 24

# 2️⃣ Encode transaction type
le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])

# -------------------------------
# Select features
# -------------------------------
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "transaction_hour",
    "type_encoded"
]

X = df[FEATURES]
y = df["isFraud"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Handle imbalance
# -------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -------------------------------
# XGBoost Model
# -------------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model & preprocessors
# -------------------------------
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/type_encoder.pkl")

print("✅ XGBoost model trained and saved successfully")
