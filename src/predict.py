import joblib
import numpy as np

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_fraud(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    return prob
