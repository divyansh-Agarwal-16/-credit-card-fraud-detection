# Credit Card Fraud Detection Pipeline — Makefile
# Usage: make <target>

.PHONY: help install install-dev train evaluate predict serve serve-prod test test-pipeline test-api lint clean clean-all docker-build docker-up docker-down

PYTHON  = python
PYTEST  = pytest
CONFIG  = configs/config.yaml

help:   ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:   ## Install all dependencies
	pip install -r requirements.txt

install-dev:   ## Install in editable mode (for development)
	pip install -e .

train:   ## Train SVM + XGBoost + LightGBM, generate SHAP plots, log experiment
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:   ## Print full evaluation report from saved models
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

predict:   ## Run a single CLI prediction (XGBoost by default)
	$(PYTHON) scripts/predict.py --model xgboost

serve:   ## Start the FastAPI server (dev mode with reload)
	$(PYTHON) scripts/serve.py --reload

serve-prod:   ## Start the FastAPI server (production, 2 workers)
	$(PYTHON) scripts/serve.py --host 0.0.0.0 --port 8000 --workers 2

test:   ## Run all tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-pipeline:   ## Run pipeline unit tests only
	$(PYTEST) tests/test_pipeline.py -v

test-api:   ## Run API endpoint tests only
	$(PYTEST) tests/test_api.py -v

lint:   ## Run flake8 linter
	flake8 src/ tests/ scripts/ --max-line-length=100 --ignore=E501,W503

clean:   ## Remove model + plot files (keeps experiment log)
	$(PYTHON) scripts/clean.py

clean-all:   ## Remove ALL generated files including experiment log
	$(PYTHON) scripts/clean.py --all

docker-build:   ## Build Docker image
	docker build -t fraud-ml-pipeline:latest .

docker-up:   ## Build and start all services via docker-compose
	docker-compose up --build

docker-down:   ## Stop all docker-compose services
	docker-compose down
