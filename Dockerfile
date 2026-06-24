# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="Divyansh Agarwal <divyanshagg296@gmail.com>"
LABEL description="Fraud Detection Pipeline — FastAPI + scikit-learn"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Source ────────────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create output directories
RUN mkdir -p outputs/models

# ── Train on build (optional — comment out for faster builds) ─────────────────
# RUN python -c "import sys; sys.path.insert(0,'src'); from pipeline import run_pipeline; run_pipeline()"

# ── Expose & Run ──────────────────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
