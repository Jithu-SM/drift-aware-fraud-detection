from src.drift_detection import detect_prediction_drift, detect_fraud_rate_drift
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
DATA_FOLDER = "data"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

TRANSACTIONS_FILE = os.path.join(DATA_FOLDER, "transactions.csv")
DRIFT_LOG_FILE = os.path.join(DATA_FOLDER, "drift_log.csv")

# -------------------------------
# Load trained artifacts
# -------------------------------
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
type_encoder = joblib.load("models/type_encoder.pkl")

REQUIRED_COLUMNS = [
    "transaction_id",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "transaction_hour",
    "type_encoded"
]

# -------------------------------
# Risk Classification
# -------------------------------
def classify_risk(prob):
    if prob > 0.8:
        return "Fraudulent 🔴", "red"
    elif prob > 0.5:
        return "Action Required 🟠", "orange"
    else:
        return "Legitimate 🟢", "green"

# -------------------------------
# Drift Logging
# -------------------------------
def log_drift(pred_drift, fraud_drift):
    drift_entry = {
        "timestamp": datetime.now(),
        "prediction_drift": pred_drift,
        "fraud_rate_drift": fraud_drift
    }

    if os.path.exists(DRIFT_LOG_FILE):
        df_log = pd.read_csv(DRIFT_LOG_FILE)
        df_log = pd.concat([df_log, pd.DataFrame([drift_entry])], ignore_index=True)
    else:
        df_log = pd.DataFrame([drift_entry])

    df_log.to_csv(DRIFT_LOG_FILE, index=False)

# -------------------------------
# Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    table_data = None
    drift_alert = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a valid CSV file.")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.stream.seek(0)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return render_template("index.html", error=str(e))

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            return render_template("index.html", error=f"Missing columns: {missing}")

        try:
            # -----------------------
            # Feature Engineering
            # -----------------------
            df["transaction_hour"] = df["step"] % 24
            df["type"] = df["type"].astype(str)
            df["type_encoded"] = type_encoder.transform(df["type"])

            X = df[FEATURES]
            X_scaled = scaler.transform(X)

            probs = model.predict_proba(X_scaled)[:, 1]
            df["fraud_probability"] = probs

            predictions = df["fraud_probability"].apply(classify_risk)
            df["fraud_prediction"] = predictions.apply(lambda x: x[0])
            df["risk_color"] = predictions.apply(lambda x: x[1])

            df["admin_label"] = None
            df["timestamp"] = datetime.now()

            # -----------------------
            # Append to transactions
            # -----------------------
            if os.path.exists(TRANSACTIONS_FILE):
                existing = pd.read_csv(TRANSACTIONS_FILE)
                combined = pd.concat([existing, df], ignore_index=True)
            else:
                combined = df.copy()

            combined.to_csv(TRANSACTIONS_FILE, index=False)

            # -----------------------
            # Run Drift Detection
            # -----------------------
            if len(combined) > 100:
                historical = combined.iloc[:-len(df)]
                recent = df

                pred_drift = detect_prediction_drift(
                    historical["fraud_probability"],
                    recent["fraud_probability"]
                )

                fraud_drift = detect_fraud_rate_drift(
                    historical["fraud_probability"] > 0.5,
                    recent["fraud_probability"] > 0.5
                )

                log_drift(pred_drift, fraud_drift)

                if pred_drift or fraud_drift:
                    drift_alert = "⚠️ Concept Drift Detected! Model retraining recommended."

            table_data = df.to_dict(orient="records")

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        table_data=table_data,
        error=error,
        drift_alert=drift_alert
    )

# -------------------------------
# Admin Override
# -------------------------------
@app.route("/override", methods=["POST"])
def override():
    transaction_id = request.form.get("transaction_id")
    new_label = request.form.get("new_label")

    if os.path.exists(TRANSACTIONS_FILE):
        df = pd.read_csv(TRANSACTIONS_FILE)
        df.loc[df["transaction_id"] == transaction_id, "admin_label"] = new_label
        df.to_csv(TRANSACTIONS_FILE, index=False)

    return redirect(url_for("view_transactions"))

# -------------------------------
# View Transactions
# -------------------------------
@app.route("/transactions")
def view_transactions():
    if os.path.exists(TRANSACTIONS_FILE):
        df = pd.read_csv(TRANSACTIONS_FILE)
        data = df.to_dict(orient="records")
    else:
        data = []

    return render_template("transactions.html", data=data)

# -------------------------------
# Download Transactions
# -------------------------------
@app.route("/download")
def download():
    return send_file(TRANSACTIONS_FILE, as_attachment=True)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)