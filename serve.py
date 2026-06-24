"""
scripts/serve.py
----------------
Launch the FastAPI inference server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080 --workers 4 --reload
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline — Start API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers (default: 1)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check models exist
    model_dir = "outputs/models"
    if not os.path.exists(model_dir) or not any(
        f.endswith(".joblib") for f in os.listdir(model_dir)
    ):
        print("⚠️   No trained models found in outputs/models/")
        print("    Run  python scripts/train.py  first, then start the server.")
        print("    Continuing anyway — /health will report degraded status.\n")

    print("=" * 60)
    print("  Fraud Detection Pipeline — API Server")
    print("=" * 60)
    print(f"  Host    : {args.host}")
    print(f"  Port    : {args.port}")
    print(f"  Workers : {args.workers}")
    print(f"  Reload  : {args.reload}")
    print(f"\n  Swagger UI → http://{args.host}:{args.port}/docs")
    print(f"  Health   → http://{args.host}:{args.port}/health")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
