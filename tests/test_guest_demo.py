import app as webapp


def test_guest_demo_is_available_without_account_and_marked_demo():
    client = webapp.app.test_client()
    response = client.get("/demo")
    assert response.status_code == 200
    assert b"Demo mode" in response.data
    assert b"Nothing you enter is saved" in response.data


def test_guest_prediction_uses_only_submitted_values_and_does_not_create_session(monkeypatch):
    called = {}

    def fake_predict(study, test, practical):
        called.update(study=study, test=test, practical=practical)
        return 63.5

    monkeypatch.setattr(webapp, "predict_current_exam", fake_predict)
    client = webapp.app.test_client()
    response = client.post("/demo", data={
        "action": "exam", "course": "DEMO 411", "study_time_hours": "8",
        "test_score": "70", "practical_score": "75", "assignment_score": "80",
    })
    assert response.status_code == 200
    assert b"Demo exam estimate" in response.data
    assert called == {"study": 8.0, "test": 70.0, "practical": 75.0}
    with client.session_transaction() as session:
        assert "user" not in session


def test_guest_demo_validates_inputs():
    response = webapp.app.test_client().post("/demo", data={
        "action": "exam", "study_time_hours": "81", "test_score": "70",
        "practical_score": "75", "assignment_score": "80",
    })
    assert response.status_code == 200
    assert b"Study time must be between 0 and 80" in response.data


def test_admin_and_lecturer_functions_remain_protected():
    client = webapp.app.test_client()
    assert client.get("/admin").status_code == 302
    assert client.get("/lecturer").status_code == 302
    assert client.post("/lecturer/upload").status_code == 302
