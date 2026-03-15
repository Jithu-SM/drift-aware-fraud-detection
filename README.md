# Bank Fraud Detection System — Adaptive XGBoost with Drift Detection

## Project layout

```
fraud_system/
├── constants.py          # PAYMENT_TYPES, DEVICE_OS (your original)
├── features.py           # FEATURES list (your original)
├── generate_datasets.py  # Creates training + drift-trigger CSVs
├── train.py              # Initial model training (your original structure)
├── encode_labels.py      # Stand-alone encoder saver (your original)
├── predict.py            # Inference + hot-swap (your original structure)
├── drift_detection.py    # 3 detectors: prediction, fraud-rate, feature
├── retrain.py            # Triggered retraining pipeline
├── transaction_store.py  # Append predictions, handle admin label flips
├── app.py                # Flask REST API
├── simulate.py           # Streams normal then drift data to demo pipeline
├── requirements.txt
├── data/
│   ├── bank_fraud_dataset.csv   (generated)
│   ├── drift_dataset.csv        (generated)
│   └── transactions.csv         (written at runtime by app.py)
└── models/
    ├── xgb_initial_model.pkl
    ├── scaler.pkl
    └── encoders.pkl
```

---

## Quick-start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate datasets
python generate_datasets.py

# 3. Train initial model
python train.py

# 4. Start API  (terminal 1)
python app.py

# 5. Run simulation  (terminal 2)
python simulate.py
```

After ~1 000 normal + 600 drift transactions the drift detectors fire,
the model auto-retrains, and the in-memory model is hot-swapped.

---

## REST API reference

### POST /predict
Score a raw transaction. Returns probability, label, and drift/retrain status.

```json
// Request body — all FEATURES as key-value pairs
{
  "intended_balcon_amount": 450.0,
  "payment_type": "CASH_OUT",
  "session_length_in_minutes": 2.1,
  "velocity_6h": 8,
  "velocity_24h": 22.0,
  "velocity_4w": 95.0,
  "credit_risk_score": 210,
  "foreign_request": 1,
  "email_is_free": 1,
  "device_fraud_count": 3,
  "device_os": "Android",
  "month": 3,
  "transaction_hour": 2,
  "is_night_transaction": 1
}

// Response
{
  "fraud_probability": 0.9231,
  "label": "Fraud",
  "auto_retrained": false,
  "drift_info": null
}
```

### GET /transactions?n=200
Returns the n most recent stored transactions (for admin dashboard).

### POST /admin/flip/<idx>
Admin flips the label on row `idx` of transactions.csv.
The corrected label is used as ground truth in the next retraining.

```json
// Response
{
  "message": "Label flipped for row 42",
  "updated_row": { ... }
}
```

### GET /drift/check?window=500
Manually run all three drift detectors.

```json
{
  "any_drift": true,
  "prediction": [true,  {"ks_statistic": 0.22, "p_value": 0.0001}],
  "fraud_rate": [true,  {"old_fraud_rate": 0.05, "new_fraud_rate": 0.31, "change": 0.26}],
  "feature":    [true,  {"drifted_features": ["velocity_6h", "credit_risk_score", ...], ...}]
}
```

### POST /drift/retrain
Manually trigger retraining regardless of drift status.

### GET /health
Basic health + counter check.

---

## Drift detection — how it works

Three independent KS-test detectors run every 100 predictions
(configurable via `CHECK_EVERY` in app.py):

| Detector | Signal | Threshold |
|---|---|---|
| Prediction drift | KS on `fraud_probability` window vs window | 0.15 |
| Fraud-rate drift | Absolute change in fraud % across windows | 0.05 |
| Feature drift | KS on 7 key input features, fires if ≥ 2 drift | 0.20 |

Any detector firing triggers retraining. Feature drift gives the
earliest warning because it operates on raw inputs before the model
output shifts.

## Admin label corrections

When an admin flips a prediction via `/admin/flip/<idx>`, the row's
`admin_label` is toggled (Fraud ↔ Legit) and `admin_corrected` is set to 1.
The retrain pipeline reads `admin_label` as ground truth, so corrections
immediately influence the next model.

## Datasets

`generate_datasets.py` produces two files:

- **bank_fraud_dataset.csv** — 10 000 rows, ~5 % fraud, normal patterns
- **drift_dataset.csv** — 3 000 rows, ~30 % fraud, shifted payment type
  distribution (CASH_OUT dominates), all night transactions, higher
  velocities and device fraud counts, lower credit scores — designed to
  trigger all three drift detectors within ~500–600 rows.