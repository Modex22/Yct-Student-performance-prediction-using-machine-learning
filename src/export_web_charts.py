"""
Export small chart-ready data (feature importances, a sample of
actual-vs-predicted test points) for the static web demo, so it can draw
its own theme-aware SVG charts instead of embedding the matplotlib PNGs.

Usage:
    python -m src.export_web_charts [output_path]
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import CLASSIFIER_PATH, REGRESSION_TARGET, REGRESSOR_PATH
from src.data_preprocessing import get_features, load_data
from src.train_model import RANDOM_STATE

FRIENDLY_NAMES = {
    "exam_score": "Exam score",
    "test_score": "Test / CA score",
    "assignment_score": "Assignment score",
    "practical_score": "Practical score",
}


def main(output_path: str = "web_demo_charts.json") -> None:
    df = load_data()
    X = get_features(df)
    y = df[REGRESSION_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = joblib.load(REGRESSOR_PATH)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        raise TypeError(f"{type(model).__name__} exposes neither feature_importances_ nor coef_")

    order = np.argsort(importances)[::-1][:8]
    feature_importance = [
        {
            "label": FRIENDLY_NAMES.get(feature_names[i].split("__", 1)[1], feature_names[i]),
            "value": round(float(importances[i]), 4),
        }
        for i in order
    ]

    predictions = pipeline.predict(X_test)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=min(150, len(X_test)), replace=False)
    scatter = [
        {"actual": round(float(y_test.iloc[i]), 1), "predicted": round(float(predictions[i]), 1)}
        for i in sample_idx
    ]

    with open(output_path, "w") as f:
        json.dump({"feature_importance": feature_importance, "actual_vs_predicted": scatter}, f)

    print(f"Wrote {output_path}: {len(feature_importance)} features, {len(scatter)} scatter points")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "web_demo_charts.json"
    main(out)
