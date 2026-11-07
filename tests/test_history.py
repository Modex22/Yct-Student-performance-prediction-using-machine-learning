import os

import pytest

from src.history import (
    WrongPin,
    build_trend_chart,
    delete_history,
    get_history,
    has_pin_set,
    save_entry,
    verify_pin,
)
from src.predict import predict_performance


@pytest.fixture(autouse=True)
def _isolated_history_db(tmp_path, monkeypatch):
    """Point HISTORY_DB_PATH at a throwaway file per test so tests never
    touch (or depend on) the real instance/history.db."""
    import src.config as config
    import src.history as history

    db_path = os.path.join(tmp_path, "history_test.db")
    monkeypatch.setattr(config, "HISTORY_DB_PATH", db_path)
    monkeypatch.setattr(history, "HISTORY_DB_PATH", db_path)
    yield


STUDENT = {"exam_score": 36, "test_score": 6, "assignment_score": 7, "practical_score": 13}


def test_save_and_get_history_round_trips():
    result = predict_performance(STUDENT)
    save_entry("ND/CS/23/0001", "1234", STUDENT, result)

    entries = get_history("ND/CS/23/0001", "1234")
    assert len(entries) == 1
    assert entries[0]["exam_score"] == 36
    assert entries[0]["predicted_score"] == result["predicted_score"]


def test_save_entry_requires_matric_no():
    result = predict_performance(STUDENT)
    with pytest.raises(ValueError):
        save_entry("", "1234", STUDENT, result)


def test_save_entry_requires_a_real_pin():
    result = predict_performance(STUDENT)
    for bad_pin in ("", "12"):
        with pytest.raises(ValueError):
            save_entry("ND/CS/23/0001", bad_pin, STUDENT, result)


def test_first_save_sets_the_pin_and_locks_out_a_different_one():
    result = predict_performance(STUDENT)
    save_entry("ND/CS/23/0002", "correct-pin", STUDENT, result)

    assert has_pin_set("ND/CS/23/0002")
    assert verify_pin("ND/CS/23/0002", "correct-pin")
    assert not verify_pin("ND/CS/23/0002", "wrong-pin")

    with pytest.raises(WrongPin):
        save_entry("ND/CS/23/0002", "wrong-pin", STUDENT, result)

    # The wrong-PIN attempt must not have appended anything.
    assert len(get_history("ND/CS/23/0002", "correct-pin")) == 1


def test_get_history_wrong_pin_raises_but_no_history_returns_empty():
    result = predict_performance(STUDENT)
    save_entry("ND/CS/23/0003", "secret", STUDENT, result)

    with pytest.raises(WrongPin):
        get_history("ND/CS/23/0003", "guess")

    # A matric number with no saved history at all is just empty, not an error
    # (nothing to unlock, so no PIN to get wrong).
    assert get_history("ND/NEVER/SAVED", "anything") == []


def test_different_matric_numbers_are_isolated():
    result = predict_performance({"exam_score": 20, "test_score": 4, "assignment_score": 4, "practical_score": 8})
    save_entry("ND/A/1", "pin-a", STUDENT, result)
    save_entry("ND/B/1", "pin-b", STUDENT, result)

    assert len(get_history("ND/A/1", "pin-a")) == 1
    assert len(get_history("ND/B/1", "pin-b")) == 1
    # A's PIN doesn't unlock B's history.
    with pytest.raises(WrongPin):
        get_history("ND/B/1", "pin-a")


def test_delete_history_requires_correct_pin_and_removes_only_that_matric_no():
    result = predict_performance(STUDENT)
    save_entry("ND/A/1", "pin-a", STUDENT, result)
    save_entry("ND/B/1", "pin-b", STUDENT, result)

    with pytest.raises(WrongPin):
        delete_history("ND/A/1", "wrong")
    assert len(get_history("ND/A/1", "pin-a")) == 1  # untouched

    deleted = delete_history("ND/A/1", "pin-a")
    assert deleted == 1
    assert get_history("ND/A/1", "pin-a") == []
    assert not has_pin_set("ND/A/1")  # PIN cleared too, so the slot can be reused
    assert len(get_history("ND/B/1", "pin-b")) == 1


def test_build_trend_chart_none_when_empty():
    assert build_trend_chart([]) is None


def test_build_trend_chart_orders_points_left_to_right():
    entries = [
        {"predicted_score": 40.0, "saved_at": "2026-01-01T00:00:00+00:00"},
        {"predicted_score": 70.0, "saved_at": "2026-02-01T00:00:00+00:00"},
    ]
    chart = build_trend_chart(entries)
    assert chart is not None
    assert len(chart["points"]) == 2
    # Higher score -> smaller y (SVG y grows downward), and time moves left to right.
    assert chart["points"][0]["cx"] < chart["points"][1]["cx"]
    assert chart["points"][1]["cy"] < chart["points"][0]["cy"]
