import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Create bank-style features
df_demo = pd.DataFrame()
df_demo["Time"] = (df["Time"] / 3600) % 24
df_demo["Amount"] = df["Amount"]

# Simulated banking risk scores
np.random.seed(42)
df_demo["LocationRisk"] = np.random.uniform(0, 1, len(df))
df_demo["AccountRisk"] = np.random.uniform(0, 1, len(df))

df_demo["HighAmount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
df_demo["LateNight"] = (df_demo["Time"] < 6).astype(int)

X = df_demo
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

print("=== MODEL EVALUATION ===")
print(classification_report(y_test, model.predict(X_test)))

# FEATURE IMPORTANCE (VERY IMPORTANT)
print("\nModel coefficients:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"{name}: {coef:.3f}")

joblib.dump(model, os.path.join(MODEL_DIR, "demo_fraud_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "demo_scaler.pkl"))

print("\nDemo model saved successfully.")
