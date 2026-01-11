from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "demo_fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "demo_scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        time = float(request.form["time"])
        amount = float(request.form["amount"])
        location_risk = float(request.form["location_risk"])
        account_risk = float(request.form["account_risk"])

        high_amount = 1 if amount > 50000 else 0
        late_night = 1 if time < 6 else 0

        input_data = np.array([[
            time,
            amount,
            location_risk,
            account_risk,
            high_amount,
            late_night
        ]])

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        result = "Fraudulent Transaction 🚨" if pred == 1 else "Legitimate Transaction ✅"
        probability = round(prob * 100, 2)

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
