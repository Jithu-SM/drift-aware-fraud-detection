from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------------------------------
# Load trained artifacts
# -------------------------------
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
type_encoder = joblib.load("models/type_encoder.pkl")

# -------------------------------
# Required columns for PaySim
# -------------------------------
REQUIRED_COLUMNS = [
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
# Home route
# -------------------------------
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

        # -------------------------------
        # Column validation
        # -------------------------------
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            error = f"Missing columns: {missing}"
            return render_template("index.html", error=error)

        try:
            # -------------------------------
            # Feature Engineering
            # -------------------------------

            # Convert step → transaction hour
            df["transaction_hour"] = df["step"] % 24

            # Encode transaction type
            df["type"] = df["type"].astype(str)
            df["type_encoded"] = type_encoder.transform(df["type"])

            # -------------------------------
            # Select features
            # -------------------------------
            X = df[FEATURES]

            # Scale
            X_scaled = scaler.transform(X)

            # -------------------------------
            # Prediction
            # -------------------------------
            probs = model.predict_proba(X_scaled)[:, 1]

            df["fraud_probability"] = probs
            df["fraud_prediction"] = np.where(
                probs >= 0.5,
                "Fraudulent 🚨",
                "Legitimate ✅"
            )

            # Save results
            result_path = os.path.join(RESULT_FOLDER, "fraud_results.csv")
            df.to_csv(result_path, index=False)

            table = df[
                ["fraud_prediction", "fraud_probability"]
            ].to_html(classes="table table-striped", index=False)

            download_file = "fraud_results.csv"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        table=table,
        download_file=download_file,
        error=error
    )

# -------------------------------
# Download route
# -------------------------------
@app.route("/download")
def download():
    return send_file(
        os.path.join(RESULT_FOLDER, "fraud_results.csv"),
        as_attachment=True
    )

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
