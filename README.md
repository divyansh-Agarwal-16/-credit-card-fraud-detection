# Credit Card Fraud Detection — ML Pipeline

End-to-end machine learning pipeline for imbalanced binary classification (fraud detection). Built as a portfolio-ready, industry-style project.

---

## What's inside

| Feature | Detail |
|---|---|
| **Real-world dataset** | Synthetic credit card fraud — imbalanced (~5% fraud), mixed types (numeric + categorical), intentional nulls (~5%) |
| **Gradient boosting** | XGBoost + LightGBM alongside SVM; tuned via GridSearchCV + StratifiedKFold |
| **SHAP** | `TreeExplainer` for XGBoost/LightGBM; beeswarm plots saved per model + per-prediction attribution via `/explain` |
| **Experiment tracking** | Append-only JSON-lines log at `outputs/experiments/runs.json`; queryable via `/experiments` endpoint |
| **Non-trivial API** | Cross-field validators (`velocity_24h > n_transactions` guard), categorical enum validation, `/explain` returns full SHAP values |

---

## Project structure

```
fraud-ml-pipeline/
├── src/
│   ├── pipeline.py         # Data · Preprocess (SMOTE) · Train · SHAP · Experiment log
│   ├── main.py             # FastAPI: /predict /batch /explain /experiments /health /metrics
│   └── __init__.py
├── scripts/
│   ├── train.py            # CLI trainer
│   ├── serve.py            # uvicorn launcher
│   ├── evaluate.py         # Print evaluation report from saved models
│   ├── predict.py          # Single CLI prediction with --model / --amount flags
│   └── clean.py            # Remove generated outputs
├── configs/
│   └── config.yaml         # All hyperparameters + SHAP + experiment settings
├── tests/
│   ├── test_pipeline.py    # Unit tests: data, preprocess, evaluate, experiment log
│   ├── test_api.py         # API tests: all endpoints + edge-case validation
│   └── __init__.py
├── outputs/
│   ├── models/             # *.joblib saved after training (gitignored)
│   ├── experiments/        # runs.json — append-only experiment log (gitignored)
│   ├── shap_xgboost.png    # generated after training
│   ├── shap_lightgbm.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: test + docker build on push
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── setup.py
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (all models + SHAP plots + experiment log)
python scripts/train.py

# 3. Serve API
python scripts/serve.py
# → Swagger UI at http://127.0.0.1:8000/docs

# 4. Test
pytest tests/ -v

# Or use make:
make install && make train && make serve
```

---

## Available make targets

```
make install        Install all dependencies
make train          Train SVM + XGBoost + LightGBM, save SHAP plots, log experiment
make evaluate       Print full evaluation report from saved models
make predict        Run a single CLI prediction (XGBoost by default)
make serve          Start API server in dev mode (hot reload)
make serve-prod     Start API server with 2 workers
make test           Run all tests with coverage
make test-pipeline  Run pipeline unit tests only
make test-api       Run API endpoint tests only
make lint           Run flake8
make clean          Remove model + plot files (keeps experiment log)
make clean-all      Remove all generated files
make docker-build   Build Docker image
make docker-up      Start via docker-compose
```

---

## API endpoints

### `POST /predict`
Single transaction fraud prediction.

```json
{
  "features": {
    "amount": 4500.0,
    "account_age_days": 120,
    "n_transactions": 8,
    "avg_spend_30d": 300.0,
    "distance_from_home": 250.0,
    "hour_of_day": 2,
    "day_of_week": 6,
    "velocity_24h": 5,
    "merchant_category": "electronics",
    "card_type": "visa"
  },
  "model": "xgboost"
}
```

Response:
```json
{
  "fraud_probability": 0.872,
  "is_fraud": true,
  "model_used": "XGBOOST",
  "threshold_used": 0.5,
  "timestamp": "2025-01-15T03:42:11"
}
```

### `POST /batch`
Up to 500 transactions in one call. Same `features` shape as `/predict`, wrapped in `"samples": [...]`.

### `POST /explain`
Returns SHAP feature attribution for one transaction (xgboost or lightgbm only):
```json
{
  "fraud_probability": 0.872,
  "is_fraud": true,
  "model_used": "XGBOOST",
  "shap_values": {
    "amount": 0.312,
    "distance_from_home": 0.198,
    "hour_of_day": 0.143,
    "velocity_24h": 0.091,
    "...": "..."
  },
  "top_fraud_drivers": ["amount (+0.3120)", "distance_from_home (+0.1980)"],
  "top_safety_factors": ["account_age_days (-0.0821)"]
}
```

### `GET /experiments?last_n=10`
Returns last N training runs from the JSON-lines experiment log.

### `GET /health` · `GET /metrics`
Standard monitoring endpoints.

---

## Input validation — real edge cases caught

| Case | HTTP |
|---|---|
| `merchant_category` not in allowed list | 422 |
| `card_type` not in allowed list | 422 |
| `velocity_24h > n_transactions` (physically impossible) | 422 |
| `amount <= 0` | 422 |
| `hour_of_day >= 24` | 422 |
| Unknown model name | 422 |
| Empty batch (`samples: []`) | 422 |

Valid `merchant_category` values: `groceries`, `electronics`, `travel`, `dining`, `entertainment`, `gas`, `retail`

Valid `card_type` values: `visa`, `mastercard`, `amex`, `discover`

Valid `model` values: `svm`, `xgboost`, `lightgbm` (for `/explain`: `xgboost` or `lightgbm` only)

---

## Key metrics (typical run)

| Model | AUC-ROC | Avg Precision | F1 |
|---|---|---|---|
| SVM | ~0.87 | ~0.62 | ~0.55 |
| XGBoost | ~0.93 | ~0.75 | ~0.68 |
| LightGBM | ~0.93 | ~0.74 | ~0.67 |

Imbalance is handled via SMOTE applied on the training split only (never leaks into test).

---

## Docker

```bash
# Build and start both trainer + API
docker-compose up --build

# API will be available at http://localhost:8000
```

The `pipeline` service trains the models and writes to `outputs/`. The `api` service starts after training completes and mounts the same `outputs/` volume.
