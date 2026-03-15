"""
cli.py  — Interactive CLI for scoring transactions
------
Test predictions, flip labels, and check drift without starting Flask.

Usage
-----
  python cli.py score              # interactive single transaction scoring
  python cli.py batch <csv>        # score every row in a CSV
  python cli.py drift              # run all drift detectors
  python cli.py retrain            # manually trigger retraining
  python cli.py stats              # print transaction store stats
"""

import sys
import json
import argparse
import pandas as pd

import predict as pred_module
from drift_detection import check_all_drift
from retrain import trigger_retrain
import transaction_store as store
from features import FEATURES
from constants import PAYMENT_TYPES, DEVICE_OS


# ── Defaults for interactive scoring ─────────────────────────────────────────
DEFAULTS = {
    "intended_balcon_amount": 200.0,
    "payment_type":           "CARD",
    "session_length_in_minutes": 5.0,
    "velocity_6h":            2.0,
    "velocity_24h":           6.0,
    "velocity_4w":            40.0,
    "credit_risk_score":      650,
    "foreign_request":        0,
    "email_is_free":          0,
    "device_fraud_count":     0,
    "device_os":              "Windows",
    "month":                  3,
    "transaction_hour":       14,
    "is_night_transaction":   0,
}


def _color(text, code): return f"\033[{code}m{text}\033[0m"
def _red(t):    return _color(t, "91")
def _green(t):  return _color(t, "92")
def _yellow(t): return _color(t, "93")
def _bold(t):   return _color(t, "1")


# ── score ─────────────────────────────────────────────────────────────────────
def cmd_score(_args):
    print(_bold("\n── Interactive transaction scorer ──"))
    print("Press Enter to use the default value shown in [brackets].\n")

    txn = {}
    for feat in FEATURES:
        default = DEFAULTS.get(feat, 0)
        raw = input(f"  {feat} [{default}]: ").strip()
        if not raw:
            txn[feat] = default
        else:
            # Try int, then float, then string
            try:   txn[feat] = int(raw)
            except ValueError:
                try:   txn[feat] = float(raw)
                except ValueError: txn[feat] = raw

    print()
    X = pred_module.preprocess(txn)
    prob = float(pred_module.predict_fraud(X)[0])
    label = "Fraud" if prob >= 0.5 else "Legit"

    bar_len = 30
    filled = int(prob * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    bar_colored = (_red if prob >= 0.5 else _green)(bar)

    print(_bold("  Result"))
    print(f"  Probability : {bar_colored}  {prob*100:.1f}%")
    print(f"  Label       : {_red(label) if label=='Fraud' else _green(label)}")

    store.append_transaction(txn, prob)
    print("  (Transaction saved to transactions.csv)\n")


# ── batch ─────────────────────────────────────────────────────────────────────
def cmd_batch(args):
    path = args.csv
    df   = pd.read_csv(path)
    print(f"Scoring {len(df)} rows from {path}…\n")

    probs  = []
    labels = []
    for _, row in df.iterrows():
        txn  = {f: row.get(f, DEFAULTS.get(f, 0)) for f in FEATURES}
        X    = pred_module.preprocess(txn)
        prob = float(pred_module.predict_fraud(X)[0])
        probs.append(prob)
        labels.append("Fraud" if prob >= 0.5 else "Legit")
        store.append_transaction(txn, prob)

    df["fraud_probability"] = probs
    df["model_label"]       = labels

    fraud_n = labels.count("Fraud")
    print(f"Done.  Fraud={fraud_n} ({fraud_n/len(df)*100:.1f}%)  Legit={len(df)-fraud_n}")

    out = path.replace(".csv", "_scored.csv")
    df.to_csv(out, index=False)
    print(f"Saved to {out}")


# ── drift ─────────────────────────────────────────────────────────────────────
def cmd_drift(_args):
    print(_bold("\n── Drift detection ──\n"))
    result = check_all_drift()

    def show(name, flag, info):
        icon = _red("DRIFT") if flag else _green("ok")
        print(f"  {name:<22} {icon}")
        if isinstance(info, dict):
            for k, v in info.items():
                if k != "details":
                    print(f"    {k}: {v}")
        print()

    show("Prediction drift",  *result["prediction"])
    show("Fraud-rate drift",  *result["fraud_rate"])
    show("Feature drift",     *result["feature"])

    if result["any_drift"]:
        print(_yellow("  ⚠  Drift detected — run `python cli.py retrain` to retrain."))
    else:
        print(_green("  ✓  Model appears stable."))
    print()


# ── retrain ───────────────────────────────────────────────────────────────────
def cmd_retrain(_args):
    print(_bold("\n── Retraining ──\n"))
    result = trigger_retrain(verbose=True)
    if "error" in result:
        print(_red(f"\n❌ {result['error']}"))
    else:
        pred_module.reload_model()
        print(_green(f"\n✅ Model hot-swapped at {result['retrain_timestamp']}"))
    print()


# ── stats ─────────────────────────────────────────────────────────────────────
def cmd_stats(_args):
    df = store.load_all()
    if df.empty:
        print("No transactions stored yet.")
        return

    print(_bold(f"\n── Transaction store ({len(df)} rows) ──\n"))
    fraud_n    = (df.get("admin_label", df.get("model_label", pd.Series())) == "Fraud").sum()
    corrected  = (df.get("admin_corrected", pd.Series(dtype=int)) == 1).sum()

    print(f"  Total rows        : {len(df):,}")
    print(f"  Fraud (admin lbl) : {fraud_n} ({fraud_n/len(df)*100:.1f}%)")
    print(f"  Admin corrections : {corrected}")
    if "fraud_probability" in df.columns:
        print(f"  Mean prob         : {df['fraud_probability'].mean():.4f}")
        print(f"  Max prob          : {df['fraud_probability'].max():.4f}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(prog="cli.py")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("score",   help="Interactive single-transaction scorer")
    batch_p = sub.add_parser("batch", help="Score all rows in a CSV")
    batch_p.add_argument("csv", help="Path to CSV file")
    sub.add_parser("drift",   help="Run drift detectors")
    sub.add_parser("retrain", help="Trigger retraining")
    sub.add_parser("stats",   help="Transaction store stats")

    args = ap.parse_args()

    dispatch = {
        "score":   cmd_score,
        "batch":   cmd_batch,
        "drift":   cmd_drift,
        "retrain": cmd_retrain,
        "stats":   cmd_stats,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()