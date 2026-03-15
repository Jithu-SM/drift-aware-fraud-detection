import joblib
import numpy as np

model = joblib.load("models/xgb_initial_model.pkl")

def predict_fraud(X_scaled):
    return model.predict_proba(X_scaled)[:, 1]
