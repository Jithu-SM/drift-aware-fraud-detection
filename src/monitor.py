"""
monitor.py  — Standalone drift monitor (no Flask needed)
----------
Run this as a cron job or background process to periodically check for
drift and retrain when detected — without going through the HTTP API.

Usage
-----
  python monitor.py                    # one-shot check
  python monitor.py --watch 60         # poll every 60 seconds
  python monitor.py --force-retrain    # skip detection, retrain immediately
"""

import argparse
import time
import datetime
import json
import os

from drift_detection import check_all_drift
from retrain import trigger_retrain
import predict as pred_module

LOG_FILE = "results/monitor.log"


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    line = f"[{_ts()}] {msg}"
    print(line)
    os.makedirs("results", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_check(window: int = 500, auto_retrain: bool = True) -> dict:
    """Run all drift detectors. Retrain if drift found. Return summary."""
    _log("Running drift check…")

    result = check_all_drift(window_size=window)

    pred_flag, pred_info = result["prediction"]
    rate_flag, rate_info = result["fraud_rate"]
    feat_flag, feat_info = result["feature"]

    _log(f"  Prediction drift : {'DETECTED' if pred_flag else 'ok'}  | {pred_info}")
    _log(f"  Fraud-rate drift : {'DETECTED' if rate_flag else 'ok'}  | {rate_info}")
    _log(f"  Feature drift    : {'DETECTED' if feat_flag else 'ok'}  | {feat_info}")

    if result["any_drift"]:
        _log("⚠️  Drift detected!")
        if auto_retrain:
            _log("Starting retrain…")
            retrain_result = trigger_retrain(verbose=False)
            if "error" in retrain_result:
                _log(f"❌ Retrain failed: {retrain_result['error']}")
            else:
                _log(f"✅ Retrain done | n={retrain_result['n_samples']} "
                     f"| fraud_rate={retrain_result['fraud_rate']:.3f} "
                     f"| F1(fraud)={retrain_result['report'].get('1', {}).get('f1-score', 0):.3f}")
                pred_module.reload_model()
        else:
            _log("Auto-retrain disabled — skipping.")
    else:
        _log("✓ No drift detected.")

    result["checked_at"] = _ts()
    return result


def main():
    ap = argparse.ArgumentParser(description="Drift monitor")
    ap.add_argument("--watch",        type=int,  default=0,
                    help="Poll interval in seconds (0 = one-shot)")
    ap.add_argument("--window",       type=int,  default=500,
                    help="Window size for KS tests")
    ap.add_argument("--no-retrain",   action="store_true",
                    help="Detect only, don't retrain")
    ap.add_argument("--force-retrain",action="store_true",
                    help="Skip detection and retrain immediately")
    args = ap.parse_args()

    if args.force_retrain:
        _log("Force-retrain requested.")
        res = trigger_retrain(verbose=True)
        if "error" not in res:
            pred_module.reload_model()
        return

    if args.watch:
        _log(f"Watching for drift every {args.watch}s (window={args.window})")
        while True:
            run_check(window=args.window, auto_retrain=not args.no_retrain)
            time.sleep(args.watch)
    else:
        result = run_check(window=args.window, auto_retrain=not args.no_retrain)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()