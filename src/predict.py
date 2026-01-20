import joblib
import numpy as np

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def normalize_probability(prob, strength=0.3):
    """
    Normalize probability to prevent values too close to 0 or 1.
    Moves probabilities toward 0.5 with strength inversely proportional to distance from extremes.
    
    Args:
        prob: Original probability (0-1)
        strength: Normalization strength (0-1), higher = more aggressive pull toward 0.5
    
    Returns:
        Normalized probability
    """
    # Map probability to (-infinity, infinity) range, then compress
    # This pulls extreme values toward 0.5
    return 0.5 + (prob - 0.5) * (1 - strength)

def predict_fraud(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]
    
    # Normalize probability to avoid extreme values
    prob = normalize_probability(prob, strength=0.3)

    return prob
