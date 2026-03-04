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
# Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    table_data = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename.endswith(".csv"):
            error = "Please upload a valid CSV file."
            return render_template("index.html", error=error)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save file properly
        file.stream.seek(0)
        file.save(filepath)

        # Make sure file exists and is not empty
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return render_template("index.html", error="Uploaded file is empty.")

        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except pd.errors.EmptyDataError:
            return render_template("index.html", error="CSV file has no readable columns.")
        except Exception as e:
            return render_template("index.html", error=str(e))
        
        
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            error = f"Missing columns: {missing}"
            return render_template("index.html", error=error)

        try:
            # Feature Engineering
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

            # Append to transactions.csv
            if os.path.exists(TRANSACTIONS_FILE):
                existing = pd.read_csv(TRANSACTIONS_FILE)
                combined = pd.concat([existing, df], ignore_index=True)
            else:
                combined = df.copy()

            combined.to_csv(TRANSACTIONS_FILE, index=False)

            table_data = df.to_dict(orient="records")

        except Exception as e:
            error = str(e)

    return render_template("index.html", table_data=table_data, error=error)

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
# View All Transactions
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
