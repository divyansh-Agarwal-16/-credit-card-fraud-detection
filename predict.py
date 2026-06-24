"""
scripts/predict.py — Run a single prediction from the CLI

Usage:
    python scripts/predict.py
    python scripts/predict.py --model xgboost
    python scripts/predict.py --model lightgbm --amount 4500 --distance 300
"""

import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Single transaction fraud prediction")
    p.add_argument("--model",            default="xgboost", choices=["svm","xgboost","lightgbm"])
    p.add_argument("--amount",           type=float, default=1250.0)
    p.add_argument("--account_age_days", type=float, default=730)
    p.add_argument("--n_transactions",   type=float, default=15)
    p.add_argument("--avg_spend_30d",    type=float, default=200.0)
    p.add_argument("--distance",         type=float, default=5.0,    dest="distance_from_home")
    p.add_argument("--hour",             type=float, default=14,     dest="hour_of_day")
    p.add_argument("--day",              type=float, default=2,      dest="day_of_week")
    p.add_argument("--velocity_24h",     type=float, default=3)
    p.add_argument("--merchant_category",default="groceries")
    p.add_argument("--card_type",        default="visa")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path("outputs/models")

    if not (model_dir / f"{args.model}.joblib").exists():
        print(f"❌  {args.model}.joblib not found. Run: python scripts/train.py")
        sys.exit(1)

    model   = joblib.load(model_dir / f"{args.model}.joblib")
    scaler  = joblib.load(model_dir / "scaler.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    encoders      = joblib.load(model_dir / "encoders.joblib")

    row = {
        "amount":             args.amount,
        "account_age_days":   args.account_age_days,
        "n_transactions":     args.n_transactions,
        "avg_spend_30d":      args.avg_spend_30d,
        "distance_from_home": args.distance_from_home,
        "hour_of_day":        args.hour_of_day,
        "day_of_week":        args.day_of_week,
        "velocity_24h":       args.velocity_24h,
        "merchant_category":  args.merchant_category,
        "card_type":          args.card_type,
    }

    for col in ["merchant_category", "card_type"]:
        row[col] = float(encoders[col].transform([row[col]])[0])

    arr   = np.array([[row[f] for f in feature_names]])
    arr_s = scaler.transform(arr)
    proba = float(model.predict_proba(arr_s)[0, 1])

    print("\n" + "=" * 45)
    print("  FRAUD PREDICTION")
    print("=" * 45)
    print(f"  Model           : {args.model.upper()}")
    print(f"  Amount          : ${args.amount:,.2f}")
    print(f"  Fraud Probability: {proba:.4f}")
    print(f"  Decision        : {'⚠️  FRAUD' if proba >= 0.5 else '✅  LEGITIMATE'}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
