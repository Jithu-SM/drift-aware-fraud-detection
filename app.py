from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
from src.predict import predict_fraud
from src.features import FEATURES

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load trained artifacts
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

REQUIRED_COLUMNS = [
    "transaction_id",
    "transaction_amount",
    "payment_type",
    "session_length_in_minutes",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "credit_risk_score",
    "foreign_request",
    "email_is_free",
    "device_fraud_count",
    "device_os",
    "month"
]

@app.route("/", methods=["GET", "POST"])
def index():
    table = None
    download_file = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename.endswith(".csv"):
            error = "Please upload a valid CSV file"
            return render_template("index.html", error=error)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # -----------------------------
        # Column validation
        # -----------------------------
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            error = f"Missing columns: {missing}"
            return render_template("index.html", error=error)

        # -----------------------------
        # Encode categoricals 
        # -----------------------------
        PAYMENT_TYPE_MAPPING = {
            "card": "AA",
            "upi": "AB",
            "transfer": "AC",
            "cash_out": "AD",
        }

        DEVICE_OS_MAPPING = {
            "android": "other",
            "ios": "other",
            "windows": "windows",
            "linux": "linux",
            "macos": "macintosh",
            "macintosh": "macintosh",
            "x11": "x11",
            "others": "other",
            "other": "other"
        }

        df["payment_type"] = df["payment_type"].astype(str).str.lower()
        df["device_os"] = df["device_os"].astype(str).str.lower()

        df["payment_type"] = df["payment_type"].map(PAYMENT_TYPE_MAPPING)
        df["device_os"] = df["device_os"].map(DEVICE_OS_MAPPING)

        # 🔒 SAFETY NET (CRITICAL)
        df["payment_type"] = df["payment_type"].fillna("INTERNET")
        df["device_os"] = df["device_os"].fillna("other")

        # if df["device_os"].isna().any():
        #     raise ValueError("Unsupported device_os detected")

        df["device_os"] = encoders["device_os"].transform(df["device_os"])
        df["payment_type"] = encoders["payment_type"].transform(df["payment_type"])


        # -----------------------------
        # Synthetic time features
        # -----------------------------
        df["transaction_hour"] = np.random.randint(0, 24, size=len(df))
        df["is_night_transaction"] = df["transaction_hour"].between(0, 5).astype(int)

        # -----------------------------
        # Feature alignment
        # -----------------------------
        df["intended_balcon_amount"] = df["transaction_amount"]
        df = df.drop(columns=["transaction_amount"])

        # -----------------------------
        # Scaling
        # -----------------------------
        X = scaler.transform(df[FEATURES])

        # -----------------------------
        # Prediction
        # -----------------------------
        probs = predict_fraud(X)
        df["fraud_probability"] = probs
        df["fraud_prediction"] = np.where(
            probs >= 0.5, "Fraudulent 🚨", "Legitimate ✅"
        )

        result_path = os.path.join(RESULT_FOLDER, "fraud_results.csv")
        df.to_csv(result_path, index=False)

        table = df[
            ["transaction_id", "fraud_prediction", "fraud_probability"]
        ].to_html(classes="table table-striped", index=False)

        download_file = "fraud_results.csv"

    return render_template(
        "index.html",
        table=table,
        download_file=download_file,
        error=error
    )

@app.route("/download")
def download():
    return send_file(
        os.path.join(RESULT_FOLDER, "fraud_results.csv"),
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)
