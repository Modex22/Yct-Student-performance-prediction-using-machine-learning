"""
Train and compare candidate models, then persist the best regressor
(predicts final exam score, 0-100) and a pass/fail classifier.

Usage:
    python -m src.train_model
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_TARGET,
    CLASSIFIER_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODELS_DIR,
    REGRESSION_TARGET,
    REGRESSOR_PATH,
)
from src.data_preprocessing import build_preprocessor, get_features, load_data

RANDOM_STATE = 42

REGRESSION_CANDIDATES = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def _ensure_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def train_regressors(X_train, X_test, y_train, y_test, preprocessor):
    """Fit each candidate regressor, cross-validate, and report test metrics."""
    results = {}
    fitted = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in REGRESSION_CANDIDATES.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results[name] = {
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "test_r2": float(r2_score(y_test, predictions)),
            "test_mae": float(mean_absolute_error(y_test, predictions)),
            "test_rmse": float(root_mean_squared_error(y_test, predictions)),
        }
        fitted[name] = pipeline
        print(f"[{name}] CV R2={cv_scores.mean():.4f} | Test R2={results[name]['test_r2']:.4f} "
              f"| MAE={results[name]['test_mae']:.2f} | RMSE={results[name]['test_rmse']:.2f}")

    best_name = max(results, key=lambda n: results[n]["test_r2"])
    return best_name, fitted, results


def train_classifier(X_train, X_test, y_train, y_test, preprocessor):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label="Pass")),
    }
    print(f"[PassFailClassifier] Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")
    return pipeline, metrics


# Matches the web app's dark/green theme (static/style.css --surface /
# --accent-strong / --border / --text / --muted) so the embedded PNGs on
# /about don't sit as a jarring white rectangle on a black page.
_FIG_BG = "#121b16"
_FIG_TEXT = "#eaf5ee"
_FIG_MUTED = "#8ca897"
_FIG_GRID = "#22322a"
_FIG_ACCENT = "#4ade80"
_FIG_RISK = "#f87171"


def _apply_dark_theme(ax) -> None:
    ax.set_facecolor(_FIG_BG)
    ax.figure.set_facecolor(_FIG_BG)
    ax.tick_params(colors=_FIG_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_FIG_GRID)
    ax.title.set_color(_FIG_TEXT)
    ax.xaxis.label.set_color(_FIG_MUTED)
    ax.yaxis.label.set_color(_FIG_MUTED)
    ax.grid(color=_FIG_GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_feature_importance(pipeline: Pipeline, model_name: str) -> None:
    """Plot per-feature importance for the winning model. Tree ensembles
    expose feature_importances_ directly; linear models don't, so their
    absolute standardized coefficients are used as the importance proxy
    instead (features are standardized upstream, so coefficients are
    already comparable). Either way this always (re)writes the file —
    the previous version silently returned for a non-tree model, leaving
    a stale plot from a prior model on disk."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_label = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
        importance_label = "|Standardized coefficient|"
    else:
        return

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    order = np.argsort(importances)[-12:]  # top 12

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in order], importances[order], color=_FIG_ACCENT)
    ax.set_xlabel(importance_label)
    ax.set_title(f"Top feature importances ({model_name})", color=_FIG_TEXT)
    _apply_dark_theme(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=120, facecolor=_FIG_BG)
    plt.close(fig)


def plot_actual_vs_predicted(y_test, predictions, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, predictions, alpha=0.55, color=_FIG_ACCENT, edgecolor="none", s=22)
    lims = [0, 100]
    ax.plot(lims, lims, color=_FIG_RISK, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual final score")
    ax.set_ylabel("Predicted final score")
    ax.set_title(f"Actual vs. predicted ({model_name})", color=_FIG_TEXT)
    _apply_dark_theme(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=120, facecolor=_FIG_BG)
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    df = load_data()
    X = get_features(df)

    y_reg = df[REGRESSION_TARGET]
    y_clf = df[CLASSIFICATION_TARGET]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
    )

    best_name, fitted_regressors, regression_results = train_regressors(
        X_train, X_test, y_reg_train, y_reg_test, build_preprocessor()
    )
    best_pipeline = fitted_regressors[best_name]
    print(f"\nBest regressor: {best_name} (test R2={regression_results[best_name]['test_r2']:.4f})")

    classifier_pipeline, classifier_metrics = train_classifier(
        X_train, X_test, y_clf_train, y_clf_test, build_preprocessor()
    )

    joblib.dump(best_pipeline, REGRESSOR_PATH)
    joblib.dump(classifier_pipeline, CLASSIFIER_PATH)

    metrics = {
        "best_regressor": best_name,
        "regression_results": regression_results,
        "classifier_metrics": classifier_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_feature_importance(best_pipeline, best_name)
    plot_actual_vs_predicted(y_reg_test, best_pipeline.predict(X_test), best_name)

    print(f"\nSaved regressor -> {REGRESSOR_PATH}")
    print(f"Saved classifier -> {CLASSIFIER_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
