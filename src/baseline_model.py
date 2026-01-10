import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluation
print(classification_report(y_test, model.predict(X_test)))

# Save model & scaler
joblib.dump(model, os.path.join(MODEL_PATH, "fraud_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

print("Model and scaler saved successfully.")
