"""Load the trained models and turn a single student's data into a prediction."""
from __future__ import annotations

import json
import os
from typing import Any

import joblib
import pandas as pd

from src.config import (
    CATEGORY_BINS,
    CATEGORY_LABELS,
    CLASSIFICATION_BINS,
    CLASSIFICATION_LABELS,
    CLASSIFIER_PATH,
    COMPONENT_MAX,
    FEATURE_COLUMNS,
    GRADE_BINS,
    GRADE_LABELS,
    METRICS_PATH,
    REGRESSOR_PATH,
)

_regressor = None
_classifier = None
_score_mae = None

# A component is flagged when it falls below this fraction of its OWN max
# mark (each component has a different max — see COMPONENT_MAX) rather
# than a single absolute threshold, since e.g. 4/10 and 24/60 represent
# the same 40% shortfall but very different raw numbers.
WEAK_THRESHOLD_FRACTION = 0.4

# Each rule: (field, condition, recommendation message, short factor phrase).
# The factor phrase feeds build_counsellor_note(); order sets priority when
# more than one component is weak (exam first, since it carries the most
# marks — see COMPONENT_MAX in src/config.py).
_RULE_FIELDS = [
    ("exam_score", "Exam", "exam performance",
     "this carries the most marks of the four components, so it has the largest effect on the final result"),
    ("test_score", "Test/CA", "test scores",
     "more consistent revision ahead of tests would help here"),
    ("practical_score", "Practical", "practical scores",
     "more engagement in lab/practical sessions would help here"),
    ("assignment_score", "Assignment", "assignment scores",
     "check assignments are being submitted complete and on time"),
]


def _make_weak_condition(field: str):
    threshold = WEAK_THRESHOLD_FRACTION * COMPONENT_MAX[field]
    return lambda v: v < threshold


RECOMMENDATION_RULES = [
    (
        field,
        _make_weak_condition(field),
        f"{label} score is below {WEAK_THRESHOLD_FRACTION * COMPONENT_MAX[field]:.0f}/{COMPONENT_MAX[field]} "
        f"({WEAK_THRESHOLD_FRACTION:.0%}) — {advice}.",
        phrase,
    )
    for field, label, phrase, advice in _RULE_FIELDS
]


def _ensure_models_trained() -> None:
    if not (os.path.exists(REGRESSOR_PATH) and os.path.exists(CLASSIFIER_PATH)):
        from src.train_model import main as train_main

        train_main()


def load_models():
    global _regressor, _classifier
    if _regressor is None or _classifier is None:
        _ensure_models_trained()
        _regressor = joblib.load(REGRESSOR_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _regressor, _classifier


def _get_score_mae() -> float | None:
    """The winning regressor's test MAE from the last training run, used as
    a plain-language error margin ("62.0, typically ±1.7") instead of
    presenting a single decimal as if it were exact. None if metrics.json
    isn't available yet (e.g. models auto-trained but metrics not saved)."""
    global _score_mae
    if _score_mae is None:
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            best = metrics["best_regressor"]
            _score_mae = metrics["regression_results"][best]["test_mae"]
        except (FileNotFoundError, KeyError):
            return None
    return _score_mae


def compute_score_breakdown(student: dict[str, Any], predicted_score: float) -> dict[str, Any]:
    """How the predicted score relates to a plain sum of the raw component
    marks. Deliberately model-agnostic (works whether the winning regressor
    is linear or a tree ensemble) rather than exposing regression
    coefficients, which would only make sense for a linear model and would
    need real feature-attribution machinery (e.g. SHAP) to be meaningful
    for a tree ensemble — not worth the dependency for four features."""
    components = [
        {"field": field, "value": student[field], "max": max_mark}
        for field, max_mark in COMPONENT_MAX.items()
    ]
    raw_sum = sum(c["value"] for c in components)
    return {
        "components": components,
        "raw_sum": round(raw_sum, 1),
        "model_adjustment": round(predicted_score - raw_sum, 1),
    }


def _to_dataframe(student: dict[str, Any]) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in student]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return pd.DataFrame([{col: student[col] for col in FEATURE_COLUMNS}])


def build_recommendations(student: dict[str, Any]) -> list[str]:
    notes = []
    for field, condition, message, _phrase in RECOMMENDATION_RULES:
        try:
            if condition(student[field]):
                notes.append(message)
        except (TypeError, KeyError):
            continue
    if not notes:
        notes.append("All key indicators look healthy — keep up the current routine.")
    return notes


def _active_factor_phrases(student: dict[str, Any]) -> list[str]:
    phrases = []
    for field, condition, _message, phrase in RECOMMENDATION_RULES:
        try:
            if condition(student[field]):
                phrases.append(phrase)
        except (TypeError, KeyError):
            continue
    return phrases


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def build_counsellor_note(student: dict[str, Any], category: str) -> str:
    """A short, report-card-style remark combining the category with the
    strongest driving factor(s), for a batch roster rather than a form of
    bullet points per student."""
    factors = _active_factor_phrases(student)

    if category == "Excellent":
        return "Outstanding trajectory — every key indicator is working in this student's favour."
    if category == "Good":
        if not factors:
            return "Consistently solid performer with no red flags in the record."
        return f"Good result overall, though {_join_phrases(factors)} is worth watching."
    if category == "Average":
        if not factors:
            return "Middling result with no single standout issue — check in periodically."
        return f"Middling result — addressing {_join_phrases(factors)} could lift this into the Good band."
    # At Risk
    if not factors:
        return "At risk despite no single flagged factor — worth a closer manual review."
    return f"At risk of failing — {_join_phrases(factors)} {'is' if len(factors) == 1 else 'are'} the biggest drag on this result."


def predict_performance(student: dict[str, Any]) -> dict[str, Any]:
    """Predict a single student's final score, letter grade, category and pass/fail odds.

    `student` must contain every key in config.FEATURE_COLUMNS.
    """
    regressor, classifier = load_models()
    X = _to_dataframe(student)

    predicted_score = float(regressor.predict(X)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

    category = str(pd.cut([predicted_score], bins=CATEGORY_BINS, labels=CATEGORY_LABELS)[0])
    grade = pd.cut([predicted_score], bins=GRADE_BINS, labels=GRADE_LABELS)[0]
    classification = pd.cut(
        [predicted_score], bins=CLASSIFICATION_BINS, labels=CLASSIFICATION_LABELS
    )[0]

    pass_fail_pred = classifier.predict(X)[0]
    proba = classifier.predict_proba(X)[0]
    pass_index = list(classifier.classes_).index("Pass")
    pass_probability = float(proba[pass_index])

    score_mae = _get_score_mae()

    return {
        "predicted_score": round(predicted_score, 1),
        "score_mae": round(score_mae, 1) if score_mae is not None else None,
        "grade": str(grade),
        "classification": str(classification),
        "performance_category": category,
        "pass_fail": pass_fail_pred,
        "pass_probability": round(pass_probability * 100, 1),
        "recommendations": build_recommendations(student),
        "counsellor_note": build_counsellor_note(student, category),
        "score_breakdown": compute_score_breakdown(student, predicted_score),
    }


if __name__ == "__main__":
    example_student = {
        "exam_score": 18,   # out of 60
        "test_score": 5,    # out of 10
        "assignment_score": 6,  # out of 10
        "practical_score": 9,   # out of 20
    }
    result = predict_performance(example_student)
    print(result)
