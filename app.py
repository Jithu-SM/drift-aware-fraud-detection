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

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
import os

import predict as pred_module
import transaction_store as store
from drift_detection import check_all_drift
from retrain import trigger_retrain

app = Flask(__name__)
CORS(app)   # allow requests from file:// and any origin (dashboard + dev)

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")

@app.route("/")
def root():
    """Redirect root to the admin dashboard."""
    return send_from_directory(DASHBOARD_DIR, "index.html")

@app.route("/dashboard")
@app.route("/dashboard/")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, "index.html")


CHECK_EVERY  = 50        # run drift check every N predictions
DRIFT_WINDOW = 200       # rows per window (needs 2x=400 rows to activate)
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
        print(f"[drift check] prediction #{_pred_counter} — running detectors (window={DRIFT_WINDOW})…")
        drift_result = check_all_drift(window_size=DRIFT_WINDOW)
        drift_info   = drift_result

        pred_ok  = drift_result["prediction"]
        rate_ok  = drift_result["fraud_rate"]
        feat_ok  = drift_result["feature"]
        print(f"  prediction drift : {pred_ok[0]} | {pred_ok[1]}")
        print(f"  fraud-rate drift : {rate_ok[0]} | {rate_ok[1]}")
        print(f"  feature drift    : {feat_ok[0]} | {feat_ok[1]}")

        if drift_result["any_drift"]:
            print(f"⚠️  Drift detected — triggering retrain…")
            retrain_result = trigger_retrain(verbose=True)
            if "error" in retrain_result:
                print(f"❌ Retrain failed: {retrain_result['error']}")
            else:
                pred_module.reload_model()
                auto_retrained = True
                print(f"✅ Retrain complete and model hot-swapped.")
        else:
            print(f"  ✓ No drift detected.")

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
        return jsonify({"error": str(e), "hint": "Use the csv_row field from /transactions, not the display position"}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    new_label = updated.get("admin_label", "?")
    print(f"[admin] Row {idx} flipped → {new_label}")
    return jsonify({
        "message":     f"Row {idx} flipped to {new_label}",
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