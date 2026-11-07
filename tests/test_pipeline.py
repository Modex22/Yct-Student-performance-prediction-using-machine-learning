import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from src.config import COMPONENT_MAX, FEATURE_COLUMNS
from src.predict import compute_score_breakdown, predict_performance


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(n_students=300, seed=1)


def test_dataset_shape_and_range(dataset):
    assert len(dataset) == 300
    assert dataset["final_score"].between(0, 100).all()
    assert set(dataset["pass_fail"].unique()) <= {"Pass", "Fail"}


def test_dataset_has_all_feature_columns(dataset):
    for col in FEATURE_COLUMNS:
        assert col in dataset.columns


def test_dataset_components_stay_within_their_own_max(dataset):
    for field, max_mark in COMPONENT_MAX.items():
        assert dataset[field].between(0, max_mark).all()


# Each on its own mark scheme (exam/60, test/10, assignment/10, practical/20).
SAMPLE_STUDENT = {
    "exam_score": 49,
    "test_score": 8,
    "assignment_score": 9,
    "practical_score": 17,
}


def test_predict_performance_returns_expected_keys():
    result = predict_performance(SAMPLE_STUDENT)
    for key in (
        "predicted_score", "grade", "classification", "performance_category",
        "pass_fail", "pass_probability", "recommendations", "counsellor_note",
    ):
        assert key in result
    assert 0 <= result["predicted_score"] <= 100
    assert result["pass_fail"] in {"Pass", "Fail"}
    assert isinstance(result["recommendations"], list) and result["recommendations"]


def test_strong_profile_scores_higher_than_weak_profile():
    weak_student = {
        "exam_score": 12, "test_score": 2, "assignment_score": 3, "practical_score": 5,
    }
    strong_result = predict_performance(SAMPLE_STUDENT)
    weak_result = predict_performance(weak_student)
    assert strong_result["predicted_score"] > weak_result["predicted_score"]


def test_component_scores_sum_to_roughly_the_final_score():
    """final_score is a straight sum of the four components (each pre-
    weighted by its own max mark) plus a little noise — so the prediction
    should land close to the raw sum, not just move in the same direction."""
    student = {"exam_score": 30, "test_score": 5, "assignment_score": 5, "practical_score": 10}
    raw_sum = sum(student.values())  # 50
    predicted = predict_performance(student)["predicted_score"]
    assert abs(predicted - raw_sum) < 5


def test_exam_score_has_the_largest_effect_on_predicted_score():
    """exam_score carries the most marks (60 of 100) so a proportionally
    equal fractional drop there should move the prediction more than the
    same fractional drop in assignment_score (10 of 100)."""
    base = {"exam_score": 48, "test_score": 8, "assignment_score": 8, "practical_score": 16}
    exam_drop = dict(base, exam_score=24)  # -50% of its own max
    assignment_drop = dict(base, assignment_score=4)  # -50% of its own max

    base_score = predict_performance(base)["predicted_score"]
    exam_drop_score = predict_performance(exam_drop)["predicted_score"]
    assignment_drop_score = predict_performance(assignment_drop)["predicted_score"]

    assert (base_score - exam_drop_score) > (base_score - assignment_drop_score)


def test_predict_performance_includes_score_mae_and_breakdown():
    result = predict_performance(SAMPLE_STUDENT)
    assert result["score_mae"] is not None and result["score_mae"] > 0
    assert "score_breakdown" in result
    assert result["score_breakdown"]["raw_sum"] == sum(SAMPLE_STUDENT.values())


def test_score_breakdown_components_match_component_max():
    breakdown = compute_score_breakdown(SAMPLE_STUDENT, predicted_score=83.0)
    fields = {c["field"]: c["max"] for c in breakdown["components"]}
    assert fields == COMPONENT_MAX
    assert breakdown["model_adjustment"] == round(83.0 - sum(SAMPLE_STUDENT.values()), 1)
