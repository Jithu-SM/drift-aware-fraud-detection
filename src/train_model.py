import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "fraud.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Encode transaction type
le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])

# Feature engineering
df["balanceDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["txHour"] = np.random.randint(0, 24, size=len(df))


features = [
    "amount",
    "type_encoded",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceDiff",
    "txHour"
]

X = df[features]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

print("=== MODEL PERFORMANCE ===")
print(classification_report(y_test, model.predict(X_test)))

# Save everything
joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "type_encoder.pkl"))

print("Model saved successfully.")
