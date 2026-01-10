import streamlit as st
import numpy as np
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

st.set_page_config(page_title="Bank Transaction Fraud Detection", layout="centered")

st.title("🏦 Bank Transaction Fraud Detection System")
st.write("Enter transaction details to check if the transaction is fraudulent.")

st.divider()

# Input fields
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

st.subheader("Encoded Transaction Features")
features = []
for i in range(1, 29):
    features.append(st.number_input(f"V{i}", value=0.0))

# Prepare input
if st.button("🔍 Check Fraud"):
    input_data = np.array([[time] + features + [amount]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction\nFraud Probability: {probability:.2f}")
