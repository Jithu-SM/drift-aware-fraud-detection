from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "models", "type_encoder.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        tx_type = request.form["type"]
        amount = float(request.form["amount"])
        old_org = float(request.form["oldbalanceOrg"])
        new_org = float(request.form["newbalanceOrig"])
        old_dest = float(request.form["oldbalanceDest"])
        new_dest = float(request.form["newbalanceDest"])

        type_encoded = encoder.transform([tx_type])[0]
        balance_diff = old_org - new_org

        input_data = np.array([[
            amount,
            type_encoded,
            old_org,
            new_org,
            old_dest,
            new_dest,
            balance_diff
        ]])

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        result = "Fraudulent 🚨" if pred == 1 else "Legitimate ✅"
        probability = round(prob * 100, 2)

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
