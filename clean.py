"""
scripts/clean.py — Remove generated outputs

Usage:
    python scripts/clean.py          # remove model + plot files only
    python scripts/clean.py --all    # remove everything under outputs/
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Clean generated outputs")
    p.add_argument("--all", action="store_true", help="Remove all outputs including experiments log")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path("outputs")

    if args.all:
        if root.exists():
            shutil.rmtree(root)
            print(f"✅ Removed {root}/")
        root.mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(exist_ok=True)
        (root / "experiments").mkdir(exist_ok=True)
        (root / "models" / ".gitkeep").touch()
        (root / "experiments" / ".gitkeep").touch()
        print("✅ Re-created empty output directories.")
        return

    # Selective clean — models and plots only, keep experiment log
    removed = []
    for pattern in ["outputs/models/*.joblib", "outputs/*.png"]:
        for f in Path().glob(pattern):
            f.unlink()
            removed.append(str(f))

    if removed:
        for r in removed:
            print(f"  removed: {r}")
    else:
        print("  Nothing to clean.")
    print("✅ Clean done (experiment log preserved).")


if __name__ == "__main__":
    main()
