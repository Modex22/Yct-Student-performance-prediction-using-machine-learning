from src.batch import BatchError, build_template_csv, parse_upload, run_batch, validate_and_prepare
from src.predict import build_counsellor_note


def test_template_csv_round_trips_through_the_full_pipeline():
    csv_text = build_template_csv()
    df = parse_upload("template.csv", csv_text.encode("utf-8"))
    df, identity_map, missing = validate_and_prepare(df)

    assert missing == []
    assert set(identity_map) == {"name", "matric_no", "department", "level"}

    outcome = run_batch(df, identity_map)
    assert outcome["summary"]["n_students"] == 3
    assert outcome["warnings"] == []
    for row in outcome["results"]:
        assert 0 <= row["predicted_score"] <= 100
        assert row["classification"] in {"Fail", "Pass", "Lower Credit", "Upper Credit", "Distinction"}
        assert row["counsellor_note"]


def test_missing_feature_columns_are_reported():
    df = parse_upload("bad.csv", b"name,age\nJohn,18\n")
    df, identity_map, missing = validate_and_prepare(df)
    assert "exam_score" in missing
    assert len(missing) > 0


def test_unsupported_file_type_raises_batch_error():
    try:
        parse_upload("notes.txt", b"hello")
        assert False, "expected BatchError"
    except BatchError:
        pass


def test_non_numeric_row_is_skipped_with_a_warning():
    csv_text = build_template_csv().replace(",47,8,9,16", ",not-a-number,8,9,16")

    parsed = parse_upload("roster.csv", csv_text.encode("utf-8"))
    parsed, identity_map, missing = validate_and_prepare(parsed)
    outcome = run_batch(parsed, identity_map)

    assert outcome["summary"]["n_students"] == 2  # one row skipped
    assert any("not-a-number" in w for w in outcome["warnings"])


def test_out_of_range_value_is_flagged_but_still_processed():
    """Catches the likely real mistake: someone enters exam_score out of
    100 instead of out of 60 (its actual max)."""
    csv_text = build_template_csv().replace(",47,8,9,16", ",95,8,9,16")

    parsed = parse_upload("roster.csv", csv_text.encode("utf-8"))
    parsed, identity_map, missing = validate_and_prepare(parsed)
    outcome = run_batch(parsed, identity_map)

    assert outcome["summary"]["n_students"] == 3  # still processed, not skipped
    assert any("outside the expected 0-60 range" in w for w in outcome["warnings"])


def test_counsellor_note_mentions_the_driving_factor_for_at_risk_students():
    student = {"exam_score": 10, "test_score": 8, "assignment_score": 8, "practical_score": 16}
    note = build_counsellor_note(student, "At Risk")
    assert "exam performance" in note.lower()


def test_counsellor_note_is_positive_for_excellent_category():
    note = build_counsellor_note({}, "Excellent")
    assert "outstanding" in note.lower()
