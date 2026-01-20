from flask import Flask, render_template, request
import numpy as np
import joblib
import os
from src.predict import predict_fraud

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load encoder (model & scaler are loaded inside predict.py)
encoder = joblib.load(os.path.join(BASE_DIR, "models", "type_encoder.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        # -------------------------------
        # Get form inputs
        # -------------------------------
        tx_type = request.form["type"]
        amount = float(request.form["amount"])
        tx_time = request.form["txTime"]       # HH:MM
        old_org = float(request.form["oldbalanceOrg"])
        old_dest = float(request.form["oldbalanceDest"])

        # -------------------------------
        # Derived values
        # -------------------------------
        new_org = old_org - amount
        new_dest = old_dest + amount

        tx_hour = int(tx_time.split(":")[0])
        type_encoded = encoder.transform([tx_type])[0]

        # -------------------------------
        # Feature vector (MUST MATCH TRAINING ORDER)
        # -------------------------------
        features = [
            amount,
            old_org,
            new_org,
            old_dest,
            new_dest,
            tx_hour,
            type_encoded
        ]

        # -------------------------------
        # ML Prediction
        # -------------------------------
        prob = predict_fraud(features)

        # -------------------------------
        # Rule-based risk adjustment
        # -------------------------------
        risk_boost = 0.0

        # Late-night transactions
        if tx_hour < 6:
            risk_boost += 0.15

        # High-risk transaction types
        if tx_type in ["TRANSFER", "CASH_OUT"]:
            risk_boost += 0.10

        # Reduce risk boost for low-risk amounts
        # Low-risk threshold: amounts below 5000
        low_risk_threshold = 5000
        if amount < low_risk_threshold:
            # Amount-based modifier: lower amounts get lower risk boost
            amount_modifier = amount / low_risk_threshold
            risk_boost *= amount_modifier

        adjusted_prob = min(prob + risk_boost, 1.0)

        final_pred = 1 if adjusted_prob >= 0.5 else 0

        result = "Fraudulent 🚨" if final_pred else "Legitimate ✅"
        probability = round(adjusted_prob * 100, 2)

    return render_template("index.html", result=result, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
