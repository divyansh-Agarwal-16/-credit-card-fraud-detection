"""
scripts/train.py — Train the Fraud Detection Pipeline

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Train Fraud Detection Pipeline")
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  Fraud Detection Pipeline — Training")
    print("=" * 65)
    results = run_pipeline(config_path=args.config)
    s = results["summary"]
    print(f"\n✅ Training complete! (run_id={s['run_id']})")
    print(f"   Best model  : {s['best_model']}")
    for model in ("SVM", "XGBoost", "LightGBM"):
        k = model.lower()
        print(f"   {model:10s}: AUC={s.get(f'{model}_auc','?')} | F1={s.get(f'{model}_f1','?')}")
    print("\n   Models   → outputs/models/")
    print("   Plots    → outputs/")
    print("   Exp log  → outputs/experiments/runs.json")


if __name__ == "__main__":
    main()
