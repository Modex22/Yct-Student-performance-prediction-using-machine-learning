"""
Export the trained regressor + classifier pipelines to a single JSON file
that a plain-JavaScript predictor can walk, with no Python/Flask backend
required. Used to build the static/client-side demo (see the artifact
published from this repo).

Usage:
    python -m src.export_web_model [output_path]
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np

from src.config import (
    CATEGORICAL_FEATURES,
    CLASSIFIER_PATH,
    NUMERIC_FEATURES,
    REGRESSOR_PATH,
)


def export_tree(tree) -> dict:
    return {
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        # value shape is (n_nodes, n_outputs, n_classes_or_1); flatten the
        # per-node leading dim we don't use (n_outputs is always 1 here).
        "value": tree.value[:, 0, :].tolist(),
    }


def export_preprocessor(preprocessor) -> dict:
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    categorical_categories = []
    if CATEGORICAL_FEATURES:
        # An encoder fit on zero columns never sets categories_ at all, so
        # only touch it when there's actually something to one-hot encode.
        onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        categorical_categories = [c.tolist() for c in onehot.categories_]
    return {
        "numeric_features": NUMERIC_FEATURES,
        "numeric_mean": scaler.mean_.tolist(),
        "numeric_scale": scaler.scale_.tolist(),
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_categories": categorical_categories,
    }


def export_regressor(pipeline) -> dict:
    """Export whichever regressor train_model.py picked as best. The
    candidate set spans two structurally different kinds of model, so the
    exported shape (and the JS code that walks it) has to branch on which
    one actually won — hardcoding one shape here silently breaks the demo
    the next time a different model type wins a retrain."""
    model = pipeline.named_steps["model"]
    preprocessor_payload = export_preprocessor(pipeline.named_steps["preprocessor"])

    if hasattr(model, "estimators_") and hasattr(model, "init_"):
        # Gradient boosting: constant init prediction + learning_rate * sum(trees).
        init_constant = float(np.ravel(model.init_.constant_)[0])
        trees = [export_tree(stage[0].tree_) for stage in model.estimators_]
        return {
            "type": "gradient_boosting_regressor",
            "init_constant": init_constant,
            "learning_rate": model.learning_rate,
            "trees": trees,
            "preprocessor": preprocessor_payload,
        }
    if hasattr(model, "estimators_"):
        # Random forest: plain average of each tree's prediction.
        trees = [export_tree(est.tree_) for est in model.estimators_]
        return {
            "type": "random_forest_regressor",
            "trees": trees,
            "preprocessor": preprocessor_payload,
        }
    if hasattr(model, "coef_"):
        # Linear / Ridge regression: a single dot product + intercept.
        intercept = model.intercept_
        return {
            "type": "linear_regression",
            "coef": np.ravel(model.coef_).tolist(),
            "intercept": float(np.ravel(intercept)[0]) if np.ndim(intercept) else float(intercept),
            "preprocessor": preprocessor_payload,
        }
    raise TypeError(f"Don't know how to export a regressor of type {type(model).__name__}")


def export_classifier(pipeline) -> dict:
    model = pipeline.named_steps["model"]
    trees = [export_tree(est.tree_) for est in model.estimators_]
    return {
        "type": "random_forest_classifier",
        "classes": model.classes_.tolist(),
        "trees": trees,
        "preprocessor": export_preprocessor(pipeline.named_steps["preprocessor"]),
    }


def main(output_path: str = "web_demo_model.json") -> None:
    regressor_pipeline = joblib.load(REGRESSOR_PATH)
    classifier_pipeline = joblib.load(CLASSIFIER_PATH)

    payload = {
        "regressor": export_regressor(regressor_pipeline),
        "classifier": export_classifier(classifier_pipeline),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f)

    import os

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Wrote {output_path} ({size_kb:.1f} KB)")
    print(f"  regressor: {payload['regressor']['type']}")
    print(f"  classifier trees: {len(payload['classifier']['trees'])}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "web_demo_model.json"
    main(out)
