"""
scripts/evaluate.py — Print full evaluation report from saved models

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/config.yaml
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from pathlib import Path
from src.pipeline import load_data, preprocess, evaluate_classifier


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved fraud detection models")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path("outputs/models")

    if not model_dir.exists() or not list(model_dir.glob("*.joblib")):
        print("❌  No models found. Run: python scripts/train.py first.")
        sys.exit(1)

    print("Loading data and preprocessing...")
    X, y = load_data()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pp = cfg.get("preprocessing", {})

    _, X_test, _, y_test, scaler, feature_names, _ = preprocess(
        X, y,
        test_size=pp.get("test_size", 0.2),
        random_state=pp.get("random_state", 42),
        use_smote=False,   # no SMOTE for evaluation split
    )

    print("\n" + "=" * 55)
    print("  EVALUATION REPORT")
    print("=" * 55)

    for name in ["svm", "xgboost", "lightgbm"]:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            print(f"  ⚠️  {name}.joblib not found, skipping.")
            continue
        model = joblib.load(path)
        res = evaluate_classifier(model, X_test, y_test, name.upper())
        print(f"\n  {name.upper()}")
        print(f"    Accuracy       : {res['accuracy']:.4f}")
        print(f"    AUC-ROC        : {res['roc_auc']:.4f}")
        print(f"    Avg Precision  : {res['average_precision']:.4f}")
        print(f"    F1 Score       : {res['f1_score']:.4f}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
