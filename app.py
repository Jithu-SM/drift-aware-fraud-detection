"""
app.py  — Flask REST API
-------
Endpoints
---------
POST /predict              → score a transaction
GET  /transactions         → list recent transactions (admin dashboard data)
POST /admin/flip/<int:idx> → admin flips label for row idx
GET  /drift/check          → run all drift detectors
POST /drift/retrain        → manually trigger retraining
GET  /health               → sanity check

Automatic drift check
---------------------
Every CHECK_EVERY predictions the app checks for drift.
If any detector fires, retraining is triggered automatically
and the in-memory model is hot-swapped.
"""

from flask import Flask, request, jsonify
import traceback

import predict as pred_module
import transaction_store as store
from drift_detection import check_all_drift
from retrain import trigger_retrain

app = Flask(__name__)

CHECK_EVERY = 100        # run drift check every N predictions
DRIFT_WINDOW = 500       # rows per window for drift detectors
_pred_counter = 0        # simple in-memory counter


# ─────────────────────────────────────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    global _pred_counter

    body = request.get_json(force=True)

    try:
        X_scaled = pred_module.preprocess(body)
        prob     = float(pred_module.predict_fraud(X_scaled)[0])
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

    store.append_transaction(body, prob)
    _pred_counter += 1

    # Automatic drift check (non-blocking: runs in same thread for simplicity)
    auto_retrained = False
    drift_info     = None
    if _pred_counter % CHECK_EVERY == 0:
        drift_result = check_all_drift(window_size=DRIFT_WINDOW)
        drift_info   = drift_result
        if drift_result["any_drift"]:
            print(f"⚠️  Drift detected after {_pred_counter} predictions — retraining…")
            trigger_retrain(verbose=True)
            pred_module.reload_model()
            auto_retrained = True

    return jsonify({
        "fraud_probability": round(prob, 4),
        "label":             "Fraud" if prob >= 0.5 else "Legit",
        "auto_retrained":    auto_retrained,
        "drift_info":        drift_info,
    })


# ─────────────────────────────────────────────────────────────────────────────
# /transactions  (admin dashboard)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/transactions", methods=["GET"])
def transactions():
    n   = int(request.args.get("n", 200))
    df  = store.get_recent(n)
    return jsonify({
        "count": len(df),
        "data":  df.to_dict(orient="records"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /admin/flip/<idx>
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/admin/flip/<int:idx>", methods=["POST"])
def admin_flip(idx):
    try:
        updated = store.flip_label(idx)
    except IndexError as e:
        return jsonify({"error": str(e)}), 404

    return jsonify({
        "message":     f"Label flipped for row {idx}",
        "updated_row": updated,
    })


# ─────────────────────────────────────────────────────────────────────────────
# /drift/check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/drift/check", methods=["GET"])
def drift_check():
    window = int(request.args.get("window", DRIFT_WINDOW))
    result = check_all_drift(window_size=window)
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# /drift/retrain
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/drift/retrain", methods=["POST"])
def manual_retrain():
    result = trigger_retrain(verbose=True)
    if "error" in result:
        return jsonify(result), 400

    pred_module.reload_model()
    return jsonify({"message": "Retrain complete", "details": result})


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "total_transactions": store.count(),
        "predictions_since_start": _pred_counter,
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)