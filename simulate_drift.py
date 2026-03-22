"""
simulate_drift.py  — Stream only the drift dataset through /predict
-----------------
Use this to quickly trigger drift detection and watch the auto-retrain
fire without waiting through 1 000 normal transactions first.

Usage
-----
  python simulate_drift.py                      # default: 600 drift rows
  python simulate_drift.py --n 300              # fewer rows
  python simulate_drift.py --api http://localhost:5000
  python simulate_drift.py --delay 0.05         # slow it down to watch live
"""

import argparse
import time
import json
import requests
import pandas as pd

from features import FEATURES

DRIFT_CSV = "data/drift_dataset.csv"
CATEGORICAL_STR_COLS = ["payment_type", "device_os"]


def _row_to_dict(row: pd.Series) -> dict:
    d = {}
    for col in FEATURES:
        if col not in row.index:
            continue
        val = row[col]
        d[col] = str(val) if col in CATEGORICAL_STR_COLS else float(val)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api",   default="http://localhost:5000")
    ap.add_argument("--n",     type=int,   default=600)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()

    # Health check
    try:
        requests.get(f"{args.api}/health", timeout=3).raise_for_status()
        print(f"✅ API is up at {args.api}")
    except Exception as e:
        print(f"❌ Cannot reach API at {args.api}: {e}")
        return

    df = pd.read_csv(DRIFT_CSV).sample(frac=1, random_state=7).head(args.n)
    print(f"\nStreaming {len(df)} drift rows → {args.api}/predict")
    print(f"{'─'*55}")

    fraud_pred = 0
    retrains   = 0
    errors     = 0

    for i, (_, row) in enumerate(df.iterrows()):
        payload = _row_to_dict(row)
        try:
            resp = requests.post(f"{args.api}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("label") == "Fraud":
                fraud_pred += 1
            if data.get("auto_retrained"):
                retrains += 1
                print(f"\n{'━'*55}")
                print(f"  🔄 AUTO-RETRAIN triggered at row {i+1}")
                di = data.get("drift_info", {})
                if di:
                    pred  = di.get("prediction", [False, {}])
                    rate  = di.get("fraud_rate", [False, {}])
                    feat  = di.get("feature",    [False, {}])
                    print(f"  Prediction drift : {'DETECTED' if pred[0] else 'ok'}  {pred[1]}")
                    print(f"  Fraud-rate drift : {'DETECTED' if rate[0] else 'ok'}  {rate[1]}")
                    print(f"  Feature drift    : {'DETECTED' if feat[0] else 'ok'}  {feat[1]}")
                print(f"{'━'*55}\n")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [!] Row {i}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i+1:>4}/{len(df)}  fraud_predicted={fraud_pred}  retrains={retrains}")

        if args.delay:
            time.sleep(args.delay)

    print(f"\n{'─'*55}")
    print(f"Done: {len(df)} rows sent | fraud={fraud_pred} | retrains={retrains} | errors={errors}")

    # Final drift check
    resp = requests.get(f"{args.api}/drift/check")
    print("\n📊 Final drift check:")
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()