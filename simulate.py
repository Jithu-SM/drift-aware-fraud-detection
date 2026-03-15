"""
simulate.py  — End-to-end pipeline demo
-----------
1. Streams 1 000 normal transactions through /predict
2. Then streams 600 drift transactions through /predict
   → drift detectors should fire and retraining should auto-trigger

Usage
-----
  python simulate.py [--api http://localhost:5000]

Run app.py in a separate terminal first.
"""

import argparse
import time
import requests
import pandas as pd
import numpy as np

NORMAL_CSV = "data/bank_fraud_dataset.csv"
DRIFT_CSV  = "data/drift_dataset.csv"

CATEGORICAL_STR_COLS = ["payment_type", "device_os"]

from features import FEATURES


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a plain dict with correct types."""
    d = {}
    for col in FEATURES:
        if col not in row.index:
            continue
        val = row[col]
        if col in CATEGORICAL_STR_COLS:
            d[col] = str(val)
        else:
            d[col] = float(val)
    return d


def stream(csv_path: str, api: str, n: int, label: str, delay: float = 0.0):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=7).head(n)  # shuffle

    fraud_pred = 0
    errors     = 0
    retrains   = 0

    print(f"\n{'─'*60}")
    print(f"Streaming {len(df):,} [{label}] rows → {api}/predict")
    print(f"{'─'*60}")

    for i, (_, row) in enumerate(df.iterrows()):
        payload = _row_to_dict(row)
        try:
            resp = requests.post(f"{api}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("label") == "Fraud":
                fraud_pred += 1
            if data.get("auto_retrained"):
                retrains += 1
                print(f"\n🔄 AUTO-RETRAIN triggered at prediction #{i+1}")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [!] Row {i}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i+1:>4}/{len(df)}  fraud_so_far={fraud_pred}  retrains={retrains}")

        if delay:
            time.sleep(delay)

    print(f"\nDone [{label}]: {len(df)} sent | fraud={fraud_pred} | retrains={retrains} | errors={errors}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api",    default="http://localhost:5000")
    ap.add_argument("--normal", type=int, default=1000)
    ap.add_argument("--drift",  type=int, default=600)
    ap.add_argument("--delay",  type=float, default=0.0)
    args = ap.parse_args()

    # Check server is up
    try:
        requests.get(f"{args.api}/health", timeout=3).raise_for_status()
        print(f"✅ API is up at {args.api}")
    except Exception as e:
        print(f"❌ Cannot reach API at {args.api}: {e}")
        return

    stream(NORMAL_CSV, args.api, args.normal, "normal", args.delay)
    stream(DRIFT_CSV,  args.api, args.drift,  "drift",  args.delay)

    # Final drift check
    resp = requests.get(f"{args.api}/drift/check")
    print("\n📊 Final drift check:")
    import json
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()