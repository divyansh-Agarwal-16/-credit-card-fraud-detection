"""
Credit Card Fraud Detection — ML Pipeline
==========================================
Dataset  : Synthetic credit card fraud (imbalanced, mixed types, nulls)
Models   : SVM · XGBoost · LightGBM
Extras   : SMOTE resampling · SHAP feature importance · experiment tracking
"""

import os
import json
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score,
)

# Optional heavy deps — degrade gracefully if absent (e.g. in CI)
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    xgb = None          # type: ignore[assignment]
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    lgb = None          # type: ignore[assignment]
    _HAS_LGB = False

try:
    import shap as shap_lib
    _HAS_SHAP = True
except ImportError:
    shap_lib = None     # type: ignore[assignment]
    _HAS_SHAP = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return cfg


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate a realistic synthetic credit card fraud dataset.
    Features: numeric (transaction amount, account age, …),
              categorical (merchant_category, card_type),
              and intentional nulls (~5% missing in 3 columns).
    Target  : is_fraud (binary, imbalanced ~15-20% fraud).
    """
    rng = np.random.default_rng(42)
    n = 10_000

    amount             = rng.lognormal(mean=4.5, sigma=1.5, size=n).clip(0.5, 25_000)
    account_age_days   = rng.integers(1, 3650, size=n).astype(float)
    n_transactions     = rng.poisson(lam=15, size=n).astype(float)
    avg_spend_30d      = rng.lognormal(mean=3.8, sigma=1.2, size=n).clip(1, 5000)
    distance_from_home = rng.exponential(scale=30, size=n)
    hour_of_day        = rng.integers(0, 24, size=n).astype(float)
    day_of_week        = rng.integers(0, 7, size=n).astype(float)
    velocity_24h       = rng.poisson(lam=3, size=n).astype(float)

    merchant_cats = ["groceries", "electronics", "travel", "dining",
                     "entertainment", "gas", "retail"]
    card_types    = ["visa", "mastercard", "amex", "discover"]
    merchant_category = rng.choice(merchant_cats, size=n)
    card_type         = rng.choice(card_types, size=n)

    df = pd.DataFrame({
        "amount":             amount,
        "account_age_days":   account_age_days,
        "n_transactions":     n_transactions,
        "avg_spend_30d":      avg_spend_30d,
        "distance_from_home": distance_from_home,
        "hour_of_day":        hour_of_day,
        "day_of_week":        day_of_week,
        "velocity_24h":       velocity_24h,
        "merchant_category":  merchant_category,
        "card_type":          card_type,
    })

    # Inject ~5% nulls into 3 columns
    for col in ["avg_spend_30d", "distance_from_home", "velocity_24h"]:
        null_idx = rng.choice(n, size=int(0.05 * n), replace=False)
        df.loc[null_idx, col] = np.nan

    # Fraud label (score-based, logistic squeeze)
    fraud_score = (
        (amount > 3000).astype(float) * 0.4 +
        (distance_from_home > 100).astype(float) * 0.3 +
        (velocity_24h > 6).astype(float) * 0.2 +
        (hour_of_day < 4).astype(float) * 0.15 +
        rng.uniform(0, 0.1, size=n)
    )
    fraud_prob = 1 / (1 + np.exp(-3 * (fraud_score - 0.6)))
    is_fraud   = (rng.uniform(size=n) < fraud_prob).astype(int)

    y = pd.Series(is_fraud, name="is_fraud")
    logger.info(
        "Dataset: %d samples | %d features | fraud rate=%.1f%% | nulls=%d",
        n, df.shape[1], y.mean() * 100, df.isnull().sum().sum(),
    )
    return df, y


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
) -> Tuple:
    """
    1. Impute nulls (median for numeric, mode for categorical)
    2. Encode categoricals with LabelEncoder
    3. Stratified train/test split
    4. StandardScaler (fit on train only — no leakage)
    5. SMOTE on train only (if available and requested)

    Returns: X_train, X_test, y_train, y_test, scaler, feature_names, encoders
    """
    X = X.copy()

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object","string"]).columns.tolist()

    # ── FIX: pandas 3.x Copy-on-Write — avoid chained inplace assignment ──
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    if use_smote and _HAS_SMOTE:
        sm = SMOTE(random_state=random_state)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        logger.info(
            "SMOTE resampled train: %d samples (fraud=%d, legit=%d)",
            X_train_s.shape[0], int(y_train.sum()), int((y_train == 0).sum()),
        )
    elif use_smote and not _HAS_SMOTE:
        logger.warning("imbalanced-learn not installed — skipping SMOTE.")

    logger.info("Train: %s | Test: %s", X_train_s.shape, X_test_s.shape)
    return X_train_s, X_test_s, y_train, y_test, scaler, feature_names, encoders


# ─── Model Training ───────────────────────────────────────────────────────────

def train_svm(X_train, y_train, param_grid=None) -> SVC:
    if param_grid is None:
        param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale"]}
    cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(probability=True, class_weight="balanced"),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    logger.info("SVM best params: %s | CV AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def train_xgboost(X_train, y_train, param_grid=None):
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw  = neg / pos if pos > 0 else 1
    base = xgb.XGBClassifier(
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if param_grid is None:
        # Remove 'scale_pos_weight' from grid (already set on estimator)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth":    [3, 5],
            "learning_rate":[0.05, 0.1],
        }
    cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info("XGBoost best params: %s | CV AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


def train_lightgbm(X_train, y_train, param_grid=None):
    if not _HAS_LGB:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    base = lgb.LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth":    [3, 5],
            "learning_rate":[0.05, 0.1],
        }
    cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info("LightGBM best params: %s | CV AUC: %.4f", grid.best_params_, grid.best_score_)
    return grid.best_estimator_


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc_v  = roc_auc_score(y_test, y_proba)
    ap     = average_precision_score(y_test, y_proba)
    f1_v   = f1_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(
        "%-10s | acc=%.4f | AUC-ROC=%.4f | Avg-Prec=%.4f | F1=%.4f",
        model_name, acc, auc_v, ap, f1_v,
    )
    return {
        "model_name":            model_name,
        "accuracy":              round(acc, 4),
        "roc_auc":               round(auc_v, 4),
        "average_precision":     round(ap, 4),
        "f1_score":              round(f1_v, 4),
        "classification_report": report,
        "confusion_matrix":      cm,
        "y_pred":                y_pred,
        "y_proba":               y_proba,
    }


# ─── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    output_dir: str = "outputs",
    max_display: int = 15,
    bg_samples: int = 100,
) -> Optional[np.ndarray]:
    """
    Compute SHAP values and save a beeswarm summary plot.
    Returns shap_values array, or None if shap is unavailable.
    """
    if not _HAS_SHAP:
        logger.warning("shap not installed — skipping SHAP computation.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    if _HAS_XGB and isinstance(model, xgb.XGBClassifier):
        explainer = shap_lib.TreeExplainer(model)
        sv        = explainer.shap_values(X_test)
    elif _HAS_LGB and isinstance(model, lgb.LGBMClassifier):
        explainer = shap_lib.TreeExplainer(model)
        sv        = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        bg        = shap_lib.sample(X_train, min(bg_samples, len(X_train)))
        explainer = shap_lib.KernelExplainer(model.predict_proba, bg)
        sv        = explainer.shap_values(X_test[:200], nsamples=50)
        if isinstance(sv, list):
            sv = sv[1]

    plt.figure(figsize=(10, 7))
    shap_lib.summary_plot(
        sv,
        X_test if not isinstance(sv, list) else X_test[:200],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    out  = f"{output_dir}/shap_{safe}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP plot → %s", out)
    return sv


def shap_values_for_sample(
    model,
    sample: np.ndarray,
    X_train: np.ndarray,
    feature_names: List[str],
    bg_samples: int = 100,
) -> Dict[str, float]:
    """Per-feature SHAP for a single sample (used by /explain endpoint)."""
    if not _HAS_SHAP:
        return {}

    if _HAS_XGB and isinstance(model, xgb.XGBClassifier):
        explainer = shap_lib.TreeExplainer(model)
        sv        = explainer.shap_values(sample)
    elif _HAS_LGB and isinstance(model, lgb.LGBMClassifier):
        explainer = shap_lib.TreeExplainer(model)
        sv        = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
    else:
        bg        = shap_lib.sample(X_train, min(bg_samples, len(X_train)))
        explainer = shap_lib.KernelExplainer(model.predict_proba, bg)
        sv        = explainer.shap_values(sample, nsamples=50)
        if isinstance(sv, list):
            sv = sv[1]

    return {name: float(val) for name, val in zip(feature_names, sv[0])}


# ─── Experiment Tracking ──────────────────────────────────────────────────────

def log_experiment(
    run_id: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    log_dir: str = "outputs/experiments",
    log_file: str = "runs.json",
) -> None:
    """Append one JSON-line record to the experiment log."""
    os.makedirs(log_dir, exist_ok=True)
    path   = os.path.join(log_dir, log_file)
    record = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(),
        "params":    params,
        "metrics":   metrics,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Experiment logged → %s (run_id=%s)", path, run_id)


# ─── Plots ────────────────────────────────────────────────────────────────────

def save_plots(results: Dict, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    names = list(results.keys())
    n     = len(names)

    # Confusion matrices
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, name in zip(axes, names):
        res = results[name]
        sns.heatmap(
            res["confusion_matrix"], annot=True, fmt="d",
            cmap="Blues", ax=ax, cbar=False,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(f"{name}\nAUC={res['roc_auc']:.3f} | F1={res['f1_score']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150)
    plt.close()
    logger.info("Saved confusion_matrices.png")

    # Multi-metric bar chart
    metric_keys   = ["roc_auc", "average_precision", "f1_score", "accuracy"]
    metric_labels = ["AUC-ROC", "Avg Precision", "F1 Score", "Accuracy"]
    x      = np.arange(len(names))
    width  = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        vals = [results[m][mk] for m in names]
        bars = ax.bar(x + i * width, vals, width, label=ml, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Multiple Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved model_comparison.png")


# ─── Model Persistence ────────────────────────────────────────────────────────

def save_models(models: Dict, scaler, output_dir: str = "outputs/models") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        path = f"{output_dir}/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, path)
        logger.info("Saved %s → %s", name, path)
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    logger.info("Saved scaler → %s/scaler.joblib", output_dir)


# ─── Master Runner ────────────────────────────────────────────────────────────

def run_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    logger.info("=" * 65)
    logger.info("CREDIT CARD FRAUD DETECTION PIPELINE — START")
    logger.info("=" * 65)

    cfg      = load_config(config_path)
    pp_cfg   = cfg.get("preprocessing", {})
    exp_cfg  = cfg.get("experiment", {})
    shap_cfg = cfg.get("shap", {})
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1 · Data
    X, y = load_data()

    # 2 · Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names, encoders = preprocess(
        X, y,
        test_size    = pp_cfg.get("test_size", 0.2),
        random_state = pp_cfg.get("random_state", 42),
        use_smote    = pp_cfg.get("smote", True),
    )

    # 3 · Train
    models: Dict[str, Any] = {}
    models["SVM"] = train_svm(X_train, y_train, cfg.get("svm", {}).get("param_grid"))

    if _HAS_XGB:
        xgb_grid = cfg.get("xgboost", {}).get("param_grid")
        # Strip non-GridSearch params from config grid
        if xgb_grid and "scale_pos_weight" in xgb_grid:
            xgb_grid = {k: v for k, v in xgb_grid.items() if k != "scale_pos_weight"}
        models["XGBoost"] = train_xgboost(X_train, y_train, xgb_grid)
    else:
        logger.warning("xgboost not installed — skipping XGBoost model.")

    if _HAS_LGB:
        lgb_grid = cfg.get("lightgbm", {}).get("param_grid")
        if lgb_grid and "is_unbalance" in lgb_grid:
            lgb_grid = {k: v for k, v in lgb_grid.items() if k != "is_unbalance"}
        models["LightGBM"] = train_lightgbm(X_train, y_train, lgb_grid)
    else:
        logger.warning("lightgbm not installed — skipping LightGBM model.")

    # 4 · Evaluate
    clf_results = {
        name: evaluate_classifier(model, X_test, y_test, name)
        for name, model in models.items()
    }

    # 5 · SHAP (tree-based models only; SVM is slow)
    shap_store: Dict[str, Any] = {}
    for name, model in models.items():
        if name in ("XGBoost", "LightGBM"):
            sv = compute_shap(
                model, X_train, X_test, feature_names, name,
                max_display=shap_cfg.get("max_display", 15),
                bg_samples =shap_cfg.get("background_samples", 100),
            )
            if sv is not None:
                shap_store[name] = sv

    # 6 · Plots
    save_plots(clf_results)

    # 7 · Persist
    out_dir = cfg.get("output", {}).get("model_dir", "outputs/models")
    save_models(models, scaler, out_dir)
    joblib.dump(feature_names, f"{out_dir}/feature_names.joblib")
    joblib.dump(encoders,      f"{out_dir}/encoders.joblib")
    joblib.dump(X_train,       f"{out_dir}/X_train_bg.joblib")
    if shap_store:
        joblib.dump(shap_store, f"{out_dir}/shap_values.joblib")

    # 8 · Experiment tracking
    best_model  = max(clf_results, key=lambda m: clf_results[m]["roc_auc"])
    exp_metrics = {
        name: {k: v for k, v in res.items()
               if k in ("accuracy", "roc_auc", "average_precision", "f1_score")}
        for name, res in clf_results.items()
    }
    log_experiment(
        run_id  = run_id,
        params  = {
            "dataset":    "synthetic_credit_fraud",
            "n_samples":  int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "smote":      pp_cfg.get("smote", True),
            "test_size":  pp_cfg.get("test_size", 0.2),
        },
        metrics  = exp_metrics,
        log_dir  = exp_cfg.get("log_dir",  "outputs/experiments"),
        log_file = exp_cfg.get("log_file", "runs.json"),
    )

    # 9 · Summary
    summary = {
        "run_id":     run_id,
        "timestamp":  datetime.now().isoformat(),
        "best_model": best_model,
        **{f"{n}_auc": r["roc_auc"]  for n, r in clf_results.items()},
        **{f"{n}_f1":  r["f1_score"] for n, r in clf_results.items()},
    }

    logger.info("=" * 65)
    logger.info("PIPELINE COMPLETE")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 65)

    return {
        "summary":            summary,
        "classifier_results": clf_results,
        "models":             models,
        "scaler":             scaler,
        "feature_names":      feature_names,
        "encoders":           encoders,
        "shap_values":        shap_store,
        "X_train":            X_train,
        "X_test":             X_test,
    }


if __name__ == "__main__":
    run_pipeline()
