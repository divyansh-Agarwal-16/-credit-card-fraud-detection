"""
Credit Card Fraud Detection — FastAPI REST API
Endpoints: /predict · /batch · /explain · /health · /metrics · /experiments
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ─── Global artefacts ─────────────────────────────────────────────────────────
MODELS:        Dict[str, Any] = {}
SCALER:        Any            = None
FEATURE_NAMES: List[str]      = []
ENCODERS:      Dict[str, Any] = {}
X_TRAIN_BG:    Any            = None

METRICS = {
    "total_predictions": 0,
    "svm_calls": 0, "xgboost_calls": 0, "lightgbm_calls": 0,
    "explain_calls": 0, "errors": 0,
}
_startup_time = datetime.now().isoformat()

VALID_MODELS        = {"svm", "xgboost", "lightgbm"}
MERCHANT_CATEGORIES = ["groceries", "electronics", "travel", "dining",
                       "entertainment", "gas", "retail"]
CARD_TYPES          = ["visa", "mastercard", "amex", "discover"]


# ─── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts on startup; nothing special on shutdown."""
    global SCALER, FEATURE_NAMES, ENCODERS, X_TRAIN_BG
    model_dir = Path("outputs/models")
    if not model_dir.exists():
        logger.warning("Model dir missing — run: python scripts/train.py first.")
    else:
        for name in ["svm", "xgboost", "lightgbm"]:
            p = model_dir / f"{name}.joblib"
            if p.exists():
                MODELS[name] = joblib.load(p)
                logger.info("Loaded %s", name.upper())

        for fname, varname in [
            ("scaler.joblib",       "SCALER"),
            ("feature_names.joblib","FEATURE_NAMES"),
            ("encoders.joblib",     "ENCODERS"),
            ("X_train_bg.joblib",   "X_TRAIN_BG"),
        ]:
            p = model_dir / fname
            if p.exists():
                val = joblib.load(p)
                globals()[varname] = val
                logger.info("Loaded %s", fname)
    yield  # app runs here


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Predict transaction fraud probability using SVM · XGBoost · LightGBM. "
        "Includes SHAP-powered /explain endpoint for per-prediction feature attribution."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    amount:             float = Field(..., gt=0, le=25_000,  example=1250.0)
    account_age_days:   float = Field(..., ge=1, le=3650,    example=730)
    n_transactions:     float = Field(..., ge=0, le=500,     example=15)
    avg_spend_30d:      float = Field(..., gt=0, le=5000,    example=200.0)
    distance_from_home: float = Field(..., ge=0, le=10_000,  example=5.0)
    hour_of_day:        float = Field(..., ge=0, lt=24,      example=14)
    day_of_week:        float = Field(..., ge=0, lt=7,       example=2)
    velocity_24h:       float = Field(..., ge=0, le=100,     example=3)
    merchant_category:  str   = Field(..., example="groceries")
    card_type:          str   = Field(..., example="visa")

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant(cls, v: str) -> str:
        if v not in MERCHANT_CATEGORIES:
            raise ValueError(f"merchant_category must be one of {MERCHANT_CATEGORIES}, got '{v}'")
        return v

    @field_validator("card_type")
    @classmethod
    def validate_card(cls, v: str) -> str:
        if v not in CARD_TYPES:
            raise ValueError(f"card_type must be one of {CARD_TYPES}, got '{v}'")
        return v

    @model_validator(mode="after")
    def cross_field_checks(self) -> "TransactionFeatures":
        if self.velocity_24h > self.n_transactions:
            raise ValueError(
                f"velocity_24h ({self.velocity_24h}) cannot exceed "
                f"n_transactions ({self.n_transactions})"
            )
        return self


def _validate_model_field(v: str, allowed: set = VALID_MODELS) -> str:
    if v not in allowed:
        raise ValueError(f"model must be one of {sorted(allowed)}, got '{v}'")
    return v


class PredictRequest(BaseModel):
    features: TransactionFeatures
    model: str = Field(default="xgboost")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        return _validate_model_field(v)


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    model_used:        str
    timestamp:         str
    threshold_used:    float = 0.5


class BatchRequest(BaseModel):
    samples: List[TransactionFeatures] = Field(..., min_length=1, max_length=500)
    model:   str = Field(default="xgboost")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        return _validate_model_field(v)


class BatchResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_used:  str
    count:       int
    timestamp:   str


class ExplainRequest(BaseModel):
    features: TransactionFeatures
    model: str = Field(default="xgboost")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        supported = {"xgboost", "lightgbm"}
        return _validate_model_field(v, supported)


class ExplainResponse(BaseModel):
    fraud_probability:  float
    is_fraud:           bool
    model_used:         str
    shap_values:        Dict[str, float]
    top_fraud_drivers:  List[str]
    top_safety_factors: List[str]
    timestamp:          str


class HealthResponse(BaseModel):
    status:        str
    models_loaded: List[str]
    timestamp:     str


class MetricsResponse(BaseModel):
    total_predictions: int
    svm_calls:         int
    xgboost_calls:     int
    lightgbm_calls:    int
    explain_calls:     int
    errors:            int
    uptime_since:      str


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _require_models(model_name: str) -> None:
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Run: python scripts/train.py")
    if model_name not in MODELS:
        raise HTTPException(503, f"Model '{model_name}' not available. Loaded: {list(MODELS)}")
    if SCALER is None:
        raise HTTPException(503, "Scaler not loaded.")
    if not FEATURE_NAMES:
        raise HTTPException(503, "Feature names not loaded.")


def _features_to_array(features: TransactionFeatures) -> np.ndarray:
    """Encode categoricals and return a (1, n_features) scaled array."""
    row: Dict[str, Any] = {
        "amount":             features.amount,
        "account_age_days":   features.account_age_days,
        "n_transactions":     features.n_transactions,
        "avg_spend_30d":      features.avg_spend_30d,
        "distance_from_home": features.distance_from_home,
        "hour_of_day":        features.hour_of_day,
        "day_of_week":        features.day_of_week,
        "velocity_24h":       features.velocity_24h,
        "merchant_category":  features.merchant_category,
        "card_type":          features.card_type,
    }
    for col in ["merchant_category", "card_type"]:
        if col in ENCODERS:
            try:
                row[col] = float(ENCODERS[col].transform([row[col]])[0])
            except ValueError:
                raise HTTPException(
                    422,
                    f"Unknown {col} value '{row[col]}'. "
                    f"Valid: {list(ENCODERS[col].classes_)}",
                )
    arr = np.array([[row[f] for f in FEATURE_NAMES]])
    return SCALER.transform(arr)


def _features_to_batch_array(samples: List[TransactionFeatures]) -> np.ndarray:
    """Stack multiple samples into (n, n_features) scaled array."""
    rows = []
    for s in samples:
        row: Dict[str, Any] = {
            "amount":             s.amount,
            "account_age_days":   s.account_age_days,
            "n_transactions":     s.n_transactions,
            "avg_spend_30d":      s.avg_spend_30d,
            "distance_from_home": s.distance_from_home,
            "hour_of_day":        s.hour_of_day,
            "day_of_week":        s.day_of_week,
            "velocity_24h":       s.velocity_24h,
            "merchant_category":  s.merchant_category,
            "card_type":          s.card_type,
        }
        for col in ["merchant_category", "card_type"]:
            if col in ENCODERS:
                row[col] = float(ENCODERS[col].transform([row[col]])[0])
        rows.append([row[f] for f in FEATURE_NAMES])
    arr = np.array(rows)
    return SCALER.transform(arr)


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "message":   "Credit Card Fraud Detection API v2",
        "docs":      "/docs",
        "health":    "/health",
        "endpoints": ["/predict", "/batch", "/explain", "/experiments", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return HealthResponse(
        status        = "ok" if MODELS else "degraded",
        models_loaded = list(MODELS.keys()),
        timestamp     = datetime.now().isoformat(),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def get_metrics():
    return MetricsResponse(**METRICS, uptime_since=_startup_time)


@app.get("/experiments", tags=["Monitoring"])
def list_experiments(last_n: int = 10):
    """Return the last N experiment runs from the append-only JSON log."""
    log_path = Path("outputs/experiments/runs.json")
    if not log_path.exists():
        return {"runs": [], "message": "No experiment log found. Run the pipeline first."}
    lines = [l.strip() for l in log_path.read_text().splitlines() if l.strip()]
    runs  = [json.loads(l) for l in lines[-last_n:]]
    return {"runs": runs, "total_runs": len(lines)}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Single transaction fraud prediction with probability score."""
    try:
        _require_models(request.model)
        arr   = _features_to_array(request.features)
        proba = float(MODELS[request.model].predict_proba(arr)[0, 1])
        METRICS["total_predictions"] += 1
        METRICS[f"{request.model}_calls"] += 1
        return PredictResponse(
            fraud_probability = round(proba, 4),
            is_fraud          = proba >= 0.5,
            model_used        = request.model.upper(),
            timestamp         = datetime.now().isoformat(),
        )
    except HTTPException:
        METRICS["errors"] += 1; raise
    except Exception as e:
        METRICS["errors"] += 1; raise HTTPException(500, str(e))


@app.post("/batch", response_model=BatchResponse, tags=["Inference"])
def batch_predict(request: BatchRequest):
    """Batch fraud prediction — up to 500 transactions."""
    try:
        _require_models(request.model)
        # ── FIX: build full (n, features) matrix before scaling ──
        arr    = _features_to_batch_array(request.samples)
        probas = MODELS[request.model].predict_proba(arr)[:, 1]
        preds  = [
            {"fraud_probability": round(float(p), 4), "is_fraud": bool(p >= 0.5)}
            for p in probas
        ]
        METRICS["total_predictions"] += len(preds)
        METRICS[f"{request.model}_calls"] += len(preds)
        return BatchResponse(
            predictions = preds,
            model_used  = request.model.upper(),
            count       = len(preds),
            timestamp   = datetime.now().isoformat(),
        )
    except HTTPException:
        METRICS["errors"] += 1; raise
    except Exception as e:
        METRICS["errors"] += 1; raise HTTPException(500, str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain(request: ExplainRequest):
    """
    SHAP feature attributions for a single transaction.
    positive shap → increases fraud probability
    negative shap → decreases fraud probability
    """
    try:
        try:
            import shap as shap_lib
            import xgboost as xgb_lib
            import lightgbm as lgb_lib
        except ImportError as e:
            raise HTTPException(503, f"Explainability requires shap/xgboost/lightgbm: {e}")

        _require_models(request.model)
        arr   = _features_to_array(request.features)
        model = MODELS[request.model]
        proba = float(model.predict_proba(arr)[0, 1])

        if isinstance(model, (xgb_lib.XGBClassifier, lgb_lib.LGBMClassifier)):
            explainer = shap_lib.TreeExplainer(model)
            sv        = explainer.shap_values(arr)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            if X_TRAIN_BG is None:
                raise HTTPException(503, "Background data for SHAP not loaded.")
            bg        = shap_lib.sample(X_TRAIN_BG, 100)
            explainer = shap_lib.KernelExplainer(model.predict_proba, bg)
            sv        = explainer.shap_values(arr, nsamples=50)
            if isinstance(sv, list):
                sv = sv[1]

        shap_dict   = {name: round(float(val), 5) for name, val in zip(FEATURE_NAMES, sv[0])}
        sorted_sv   = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        top_fraud   = [f"{k} (+{v:.4f})" for k, v in sorted_sv[:3]    if v > 0]
        top_safety  = [f"{k} ({v:.4f})"  for k, v in reversed(sorted_sv) if v < 0][:3]

        METRICS["explain_calls"]      += 1
        METRICS["total_predictions"]  += 1
        METRICS[f"{request.model}_calls"] += 1

        return ExplainResponse(
            fraud_probability  = round(proba, 4),
            is_fraud           = proba >= 0.5,
            model_used         = request.model.upper(),
            shap_values        = shap_dict,
            top_fraud_drivers  = top_fraud,
            top_safety_factors = top_safety,
            timestamp          = datetime.now().isoformat(),
        )
    except HTTPException:
        METRICS["errors"] += 1; raise
    except Exception as e:
        METRICS["errors"] += 1; raise HTTPException(500, str(e))
