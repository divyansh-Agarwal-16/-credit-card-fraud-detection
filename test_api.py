"""
Tests for FastAPI endpoints — Fraud Detection API
Run: pytest tests/test_api.py -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Import app directly (not through __init__ which no longer exports it)
from src.main import app
import src.main as m

client = TestClient(app)

SAMPLE_TX = {
    "amount":             1250.0,
    "account_age_days":   730,
    "n_transactions":     15,
    "avg_spend_30d":      200.0,
    "distance_from_home": 5.0,
    "hour_of_day":        14,
    "day_of_week":        2,
    "velocity_24h":       3,
    "merchant_category":  "groceries",
    "card_type":          "visa",
}


@pytest.fixture(autouse=True)
def mock_ml_state(monkeypatch):
    """Inject mock models, scaler, encoders, feature names into app globals."""
    from sklearn.preprocessing import LabelEncoder

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    # ── FIX: scaler.transform must return (n, 10) matching input rows ──
    def smart_transform(X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.zeros((n, 10))

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = smart_transform

    feature_names = [
        "amount", "account_age_days", "n_transactions", "avg_spend_30d",
        "distance_from_home", "hour_of_day", "day_of_week", "velocity_24h",
        "merchant_category", "card_type",
    ]

    enc_mc = LabelEncoder().fit(["groceries", "electronics", "travel",
                                  "dining", "entertainment", "gas", "retail"])
    enc_ct = LabelEncoder().fit(["visa", "mastercard", "amex", "discover"])
    encoders = {"merchant_category": enc_mc, "card_type": enc_ct}

    monkeypatch.setattr(m, "MODELS",        {"svm": mock_model, "xgboost": mock_model, "lightgbm": mock_model})
    monkeypatch.setattr(m, "SCALER",        mock_scaler)
    monkeypatch.setattr(m, "FEATURE_NAMES", feature_names)
    monkeypatch.setattr(m, "ENCODERS",      encoders)


# ─── Health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_field(self):
        data = client.get("/health").json()
        assert data["status"] in ("ok", "degraded")

    def test_models_loaded_list(self):
        data = client.get("/health").json()
        assert isinstance(data["models_loaded"], list)


# ─── Root ─────────────────────────────────────────────────────────────────────
class TestRoot:
    def test_returns_200(self):
        assert client.get("/").status_code == 200

    def test_has_docs_link(self):
        assert "docs" in client.get("/").json()

    def test_endpoints_listed(self):
        data = client.get("/").json()
        assert "/explain" in data.get("endpoints", [])


# ─── Predict ──────────────────────────────────────────────────────────────────
class TestPredict:
    def test_returns_200(self):
        resp = client.post("/predict", json={"features": SAMPLE_TX, "model": "xgboost"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self):
        data = client.post("/predict", json={"features": SAMPLE_TX, "model": "xgboost"}).json()
        for f in ("fraud_probability", "is_fraud", "model_used", "timestamp"):
            assert f in data

    def test_fraud_probability_in_range(self):
        data = client.post("/predict", json={"features": SAMPLE_TX, "model": "xgboost"}).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_is_fraud_is_bool(self):
        data = client.post("/predict", json={"features": SAMPLE_TX, "model": "xgboost"}).json()
        assert isinstance(data["is_fraud"], bool)

    def test_invalid_model_returns_422(self):
        resp = client.post("/predict", json={"features": SAMPLE_TX, "model": "random_forest"})
        assert resp.status_code == 422

    def test_invalid_merchant_returns_422(self):
        bad  = {**SAMPLE_TX, "merchant_category": "casino"}
        resp = client.post("/predict", json={"features": bad, "model": "xgboost"})
        assert resp.status_code == 422

    def test_invalid_card_type_returns_422(self):
        bad  = {**SAMPLE_TX, "card_type": "diners"}
        resp = client.post("/predict", json={"features": bad, "model": "xgboost"})
        assert resp.status_code == 422

    def test_velocity_exceeds_n_transactions_returns_422(self):
        bad  = {**SAMPLE_TX, "velocity_24h": 50, "n_transactions": 10}
        resp = client.post("/predict", json={"features": bad, "model": "xgboost"})
        assert resp.status_code == 422

    def test_negative_amount_returns_422(self):
        bad  = {**SAMPLE_TX, "amount": -100}
        resp = client.post("/predict", json={"features": bad, "model": "xgboost"})
        assert resp.status_code == 422

    def test_svm_model_works(self):
        resp = client.post("/predict", json={"features": SAMPLE_TX, "model": "svm"})
        assert resp.status_code == 200
        assert resp.json()["model_used"] == "SVM"


# ─── Batch ────────────────────────────────────────────────────────────────────
class TestBatch:
    def test_returns_200(self):
        m.MODELS["xgboost"].predict_proba.return_value = np.array([[0.9, 0.1]] * 3)
        resp = client.post("/batch", json={"samples": [SAMPLE_TX] * 3, "model": "xgboost"})
        assert resp.status_code == 200

    def test_count_matches_input(self):
        n_samples = 4
        m.MODELS["xgboost"].predict_proba.return_value = np.array([[0.9, 0.1]] * n_samples)
        data = client.post("/batch", json={"samples": [SAMPLE_TX] * n_samples, "model": "xgboost"}).json()
        assert data["count"] == n_samples

    def test_predictions_length_matches(self):
        n_samples = 5
        m.MODELS["xgboost"].predict_proba.return_value = np.array([[0.9, 0.1]] * n_samples)
        data = client.post("/batch", json={"samples": [SAMPLE_TX] * n_samples, "model": "xgboost"}).json()
        assert len(data["predictions"]) == n_samples

    def test_empty_samples_returns_422(self):
        resp = client.post("/batch", json={"samples": [], "model": "xgboost"})
        assert resp.status_code == 422

    def test_each_prediction_has_fraud_probability(self):
        m.MODELS["xgboost"].predict_proba.return_value = np.array([[0.9, 0.1]] * 2)
        data = client.post("/batch", json={"samples": [SAMPLE_TX, SAMPLE_TX], "model": "xgboost"}).json()
        for pred in data["predictions"]:
            assert "fraud_probability" in pred
            assert "is_fraud" in pred


# ─── Metrics ──────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_returns_200(self):
        assert client.get("/metrics").status_code == 200

    def test_has_total_predictions(self):
        data = client.get("/metrics").json()
        assert "total_predictions" in data
        assert "uptime_since" in data


# ─── Experiments ──────────────────────────────────────────────────────────────
class TestExperiments:
    def test_returns_200(self):
        assert client.get("/experiments").status_code == 200

    def test_has_runs_field(self):
        data = client.get("/experiments").json()
        assert "runs" in data
