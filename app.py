"""Flask web app for the Yabatech Student Performance Predictor."""
from __future__ import annotations

import json
import os
import io
import pandas as pd

from flask import Flask, Response, render_template, request, send_from_directory, redirect, url_for, session, flash

from src.batch import BatchError, build_template_csv, parse_upload, run_batch, validate_and_prepare
from src.config import FEATURE_COLUMNS, FIGURES_DIR, METRICS_PATH
from src.history import WrongPin, build_trend_chart, delete_history, get_history, save_entry
from src.predict import predict_performance
from src.accounts import init_db, create_user, authenticate, save_scores, get_scores, admin_stats, get_students, log_activity, log_prediction, admin_usage
from src.semester_training import train_transition_model, template_csv, predict_second_semester
from src.current_exam_training import train_current_exam_model, exam_template_csv, predict_current_exam, validate_exam_frame
from src.accounts import save_lecturer_results, get_lecturer_results

app = Flask(__name__)
app.secret_key = __import__("os").environ.get("SECRET_KEY", "dev-only-change-this-secret-before-deployment")
init_db()
COURSES = ["COM 411", "COM 412", "COM 413", "COM 414", "COM 415", "COM 416", "COM 417", "COM 418", "COM 401", "COM 301", "COM 311", "COM 314", "COM 317", "GNS 301", "STA 301", "STA 314"]

def current_user():
    return session.get("user")

def role_required(role):
    def decorator(fn):
        from functools import wraps
        @wraps(fn)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                return redirect(url_for("login", next=request.path))
            if user.get("role") != role:
                return ("Forbidden", 403)
            return fn(*args, **kwargs)
        return wrapped
    return decorator

FORM_FIELDS = [
    {"name": "exam_score", "label": "Exam score", "type": "number", "min": 0, "max": 60, "default": 36,
     "help": "Out of 60."},
    {"name": "test_score", "label": "Test / CA score", "type": "number", "min": 0, "max": 10, "default": 6,
     "help": "Out of 10."},
    {"name": "assignment_score", "label": "Assignment score", "type": "number", "min": 0, "max": 10, "default": 7,
     "help": "Out of 10."},
    {"name": "practical_score", "label": "Practical score", "type": "number", "min": 0, "max": 20, "default": 13,
     "help": "Out of 20."},
]

NUMERIC_FIELD_NAMES = {f["name"] for f in FORM_FIELDS if f["type"] == "number"}


def parse_form(form) -> dict:
    student = {}
    for field in FORM_FIELDS:
        raw = form.get(field["name"])
        if field["name"] in NUMERIC_FIELD_NAMES:
            student[field["name"]] = float(raw)
        else:
            student[field["name"]] = raw
    return student


@app.route("/", methods=["GET"])
def index():
    user = current_user()
    if not user:
        return render_template("landing.html")
    if user.get("role") == "student":
        return redirect(url_for("student_dashboard"))
    if user.get("role") == "lecturer":
        return redirect(url_for("lecturer_dashboard"))
    return render_template("batch_upload.html", stats=admin_stats(), user=user)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name=(request.form.get("full_name") or "").strip()
        matric=(request.form.get("matric_no") or "").strip()
        password=request.form.get("password") or ""
        if not name or not matric or len(password)<8:
            return render_template("auth.html", title="Create student account", register=True, error="Enter your name, matric number and a password of at least 8 characters."), 400
        if password != request.form.get("confirm_password"):
            return render_template("auth.html", title="Create student account", register=True, error="Passwords do not match."), 400
        if not create_user(matric,name,password):
            return render_template("auth.html", title="Create student account", register=True, error="That matric number already has an account."), 409
        user=authenticate(matric,password); session["user"]={"id":user["id"],"matric_no":user["matric_no"],"full_name":user["full_name"],"role":user["role"]}
        log_activity(user["id"], "registration", {"role": user["role"]})
        return redirect(url_for("student_dashboard"))
    return render_template("auth.html", title="Create student account", register=True)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user=authenticate(request.form.get("matric_no", ""),request.form.get("password", ""))
        if not user:
            log_activity(None, "login_failed", {"identifier": (request.form.get("matric_no", "")[:32])})
            return render_template("auth.html", title="Log in", error="Incorrect username/matric number or password."), 401
        session.clear(); session["user"]={"id":user["id"],"matric_no":user["matric_no"],"full_name":user["full_name"],"role":user["role"]}
        log_activity(user["id"], "login_success", {"role": user["role"]})
        if user["role"] == "lecturer": return redirect(url_for("lecturer_dashboard"))
        if user["role"] == "admin": return redirect(url_for("admin_dashboard"))
        return redirect(url_for("student_dashboard"))
    return render_template("auth.html", title="Log in")

@app.route("/logout")
def logout():
    user=current_user()
    if user: log_activity(user["id"], "logout")
    session.clear(); return redirect(url_for("login"))

@app.route("/student")
@role_required("student")
def student_dashboard():
    user=current_user(); return render_template("student_dashboard.html",user=user,entries=get_scores(user["id"]), published_results=get_lecturer_results(user["matric_no"]))

@app.route("/student/scores", methods=["GET","POST"])
@role_required("student")
def student_scores():
    values={}; error=None
    if request.method == "POST":
        scores={}
        try:
            for course in COURSES:
                raw=(request.form.get(course) or "").strip()
                if raw:
                    val=float(raw)
                    if not 0 <= val <= 100: raise ValueError(f"{course} must be between 0 and 100.")
                    scores[course]=round(val,2)
            if len(scores)<2: raise ValueError("Enter scores for at least two courses.")
        except ValueError as exc:
            error=str(exc); values=request.form
        else:
            average=round(sum(scores.values())/len(scores),2)
            save_scores(current_user()["id"],scores,average)
            log_activity(current_user()["id"], "score_submission", {"course_count": len(scores)})
            log_prediction(current_user()["id"], "semester_one_analysis")
            category="Excellent" if average>=70 else "Good" if average>=60 else "Average" if average>=50 else "Needs support"
            return render_template("student_analysis.html",scores=scores,average=average,category=category)
    return render_template("student_scores.html",courses=COURSES,values=values,error=error)

@app.route("/student/course-predict", methods=["GET", "POST"])
@role_required("student")
def student_course_predict():
    result = None; error = None
    if request.method == "POST":
        try:
            course=(request.form.get("course") or "").strip().upper()
            study=float(request.form.get("study_time_hours", "")); assignment=float(request.form.get("assignment_score", "")); practical=float(request.form.get("practical_score", "")); test=float(request.form.get("test_score", ""))
            if not course: raise ValueError("Enter the course code.")
            if not 0 <= study <= 80: raise ValueError("Study time must be between 0 and 80 hours per week.")
            if any(not 0 <= x <= 100 for x in [assignment,practical,test]): raise ValueError("Assignment, practical and test scores must be between 0 and 100.")
            exam=predict_current_exam(study,test,practical)
            # A transparent course estimate; assignment is included in the coursework average, not the exam model.
            course_estimate=round((assignment+practical+test+exam)/4,1)
            result={"course":course,"exam":exam,"course_estimate":course_estimate,"study":study,"assignment":assignment,"practical":practical,"test":test}
            log_prediction(current_user()["id"],"current_exam_estimate"); log_activity(current_user()["id"],"current_exam_estimate",{"course":course})
        except ValueError as exc: error=str(exc)
    return render_template("student_course_predict.html",result=result,error=error)

@app.route("/student/next-semester", methods=["POST"])
@role_required("student")
def student_next_semester():
    try:
        course=(request.form.get("course") or "").strip().upper(); score=float(request.form.get("first_semester_score", ""))
        if not course or not 0 <= score <= 100: raise ValueError("Enter a course and a first-semester score from 0 to 100.")
        prediction=predict_second_semester(course,score)
        log_prediction(current_user()["id"],"next_semester_course_prediction")
        return render_template("next_semester_result.html",course=course,score=score,prediction=prediction)
    except ValueError as exc:
        return render_template("student_course_predict.html",error=str(exc),result=None),400

@app.route("/lecturer/login", methods=["GET","POST"])
def lecturer_login():
    if request.method == "POST":
        user=authenticate(request.form.get("username", ""),request.form.get("password", ""))
        if not user or user.get("role") != "lecturer":
            return render_template("lecturer_login.html",error="Invalid lecturer username or password."),401
        session.clear(); session["user"]={"id":user["id"],"matric_no":user["matric_no"],"full_name":user["full_name"],"role":user["role"]}
        log_activity(user["id"],"lecturer_login"); return redirect(url_for("lecturer_dashboard"))
    return render_template("lecturer_login.html")

@app.route("/lecturer")
@role_required("lecturer")
def lecturer_dashboard():
    return render_template("lecturer_dashboard.html",user=current_user(),results=get_lecturer_results())

@app.route("/lecturer/upload",methods=["POST"])
@role_required("lecturer")
def lecturer_upload():
    upload=request.files.get("spreadsheet")
    if not upload or not upload.filename: flash("Choose a CSV or Excel spreadsheet first.","error"); return redirect(url_for("lecturer_dashboard"))
    try:
        filename=upload.filename.lower(); data=upload.read()
        if filename.endswith('.csv'): frame=pd.read_csv(io.BytesIO(data))
        elif filename.endswith(('.xlsx','.xlsm')): frame=pd.read_excel(io.BytesIO(data),engine='openpyxl')
        else: raise ValueError("Upload a CSV or XLSX spreadsheet.")
        # Accept matric_no as the student identifier used by the portal.
        frame.columns=[str(c).strip().lower().replace(' ','_') for c in frame.columns]
        if 'matric_no' in frame.columns and 'student_id' not in frame.columns: frame=frame.rename(columns={'matric_no':'student_id'})
        frame=validate_exam_frame(frame)
        rows=[]
        for _,r in frame.iterrows(): rows.append({'matric_no':r.student_id,'course':r.course,'study_time_hours':float(r.study_time_hours),'assignment_score':None,'practical_score':float(r.practical_score),'test_score':float(r.test_score),'exam_score':float(r.exam_score)})
        # Preserve assignment scores when supplied, although they are not an input to the exam estimator.
        raw=pd.read_csv(io.BytesIO(data)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(data),engine='openpyxl')
        raw.columns=[str(c).strip().lower().replace(' ','_') for c in raw.columns]
        if 'assignment_score' in raw.columns:
            for row,(_,rr) in zip(rows,raw.iterrows()): row['assignment_score']=float(rr['assignment_score']) if pd.notna(rr['assignment_score']) else None
        metrics=train_current_exam_model(frame)
        save_lecturer_results(rows,current_user()['id'])
        log_activity(current_user()['id'],'lecturer_spreadsheet_uploaded',{'rows':len(rows),'students':metrics['students']})
        flash(f"Uploaded {len(rows)} course records and trained the exam estimator using {metrics['students']} students.","success")
    except Exception as exc: flash(str(exc),"error")
    return redirect(url_for("lecturer_dashboard"))

@app.route("/lecturer/template")
@role_required("lecturer")
def lecturer_template():
    return Response(exam_template_csv(),mimetype="text/csv",headers={"Content-Disposition":"attachment; filename=lecturer_assessment_template.csv"})

@app.route("/admin")
@role_required("admin")
def admin_dashboard():
    return render_template("admin_dashboard.html",stats=admin_stats(),usage=admin_usage(),user=current_user(),students=get_students())

@app.route("/admin/training", methods=["GET", "POST"])
@role_required("admin")
def admin_training():
    metrics = None
    error = None
    if request.method == "POST":
        upload = request.files.get("training_file")
        if not upload or not upload.filename:
            error = "Choose a CSV or Excel file first."
        else:
            try:
                name = upload.filename.lower()
                if name.endswith(".csv"):
                    frame = pd.read_csv(io.BytesIO(upload.read()))
                elif name.endswith((".xlsx", ".xlsm")):
                    frame = pd.read_excel(io.BytesIO(upload.read()), engine="openpyxl")
                else:
                    raise ValueError("Upload a .csv or .xlsx file.")
                metrics = train_transition_model(frame)
                log_activity(current_user()["id"], "semester_model_trained", {"rows": metrics["rows"], "students": metrics["students"], "best_model": metrics["best_model"]})
                flash("Semester transition model trained and saved successfully.", "success")
            except Exception as exc:
                error = str(exc)
    try:
        with open(os.path.join(app.root_path, "models", "semester_transition_metrics.json")) as f:
            saved_metrics = json.load(f)
    except (OSError, ValueError):
        saved_metrics = None
    return render_template("admin_training.html", user=current_user(), metrics=metrics, saved_metrics=saved_metrics, error=error)

@app.route("/admin/training/template")
@role_required("admin")
def admin_training_template():
    return Response(template_csv(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=semester_transition_training_template.csv"})

@app.route("/admin/students")
@role_required("admin")
def admin_students():
    return render_template("admin_students.html", students=get_students(), user=current_user())

@app.route("/batch/template", methods=["GET"])
def batch_template():
    csv_text = build_template_csv()
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=yabatech_roster_template.csv"},
    )


@app.route("/batch/predict", methods=["POST"])
@role_required("admin")
def batch_predict():
    upload = request.files.get("roster")
    if upload is None or upload.filename == "":
        return render_template("batch_error.html", message="No file was uploaded."), 400

    try:
        df = parse_upload(upload.filename, upload.read())
        df, identity_map, missing = validate_and_prepare(df)
    except BatchError as exc:
        return render_template("batch_error.html", message=str(exc)), 400

    if missing:
        return render_template(
            "batch_error.html",
            message=f"The file is missing {len(missing)} required column(s): {', '.join(missing)}.",
            missing=missing,
            all_columns=FEATURE_COLUMNS,
        ), 400

    outcome = run_batch(df, identity_map)
    return render_template("batch_results_fragment.html", outcome=outcome)


@app.route("/quick-check", methods=["GET"])
@role_required("student")
def quick_check():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/quick-check/predict", methods=["POST"])
@role_required("student")
def quick_check_predict():
    student = parse_form(request.form)
    result = predict_performance(student)
    log_prediction(current_user()["id"], "quick_check")
    log_activity(current_user()["id"], "quick_check")

    save_to_history = request.form.get("save_to_history") == "on"
    matric_no = (request.form.get("matric_no") or "").strip()
    pin = request.form.get("pin") or ""
    saved = False
    save_error = None
    trend = None

    if save_to_history:
        if not matric_no:
            save_error = "Matric number is required to save to history."
        elif not pin or len(pin) < 4:
            save_error = "A PIN of at least 4 characters is required to save to history."
        else:
            try:
                save_entry(matric_no, pin, student, result)
                saved = True
                trend = build_trend_chart(get_history(matric_no, pin))
            except WrongPin:
                save_error = (
                    f"That PIN doesn't match the one already set for {matric_no}. "
                    "If this is your first time saving, double-check the matric number for typos."
                )

    return render_template(
        "result.html",
        student=student,
        fields=FORM_FIELDS,
        result=result,
        matric_no=matric_no,
        saved=saved,
        save_error=save_error,
        trend=trend,
    )


@app.route("/history", methods=["GET"])
def history_lookup():
    return render_template("history_lookup.html")


@app.route("/history", methods=["POST"])
def history_lookup_submit():
    matric_no = (request.form.get("matric_no") or "").strip()
    pin = request.form.get("pin") or ""
    if not matric_no or not pin:
        return render_template("history_lookup.html", error="Enter both a matric number and its PIN.")

    try:
        entries = get_history(matric_no, pin)
    except WrongPin:
        return render_template("history_lookup.html", error="Incorrect PIN for that matric number.")

    if not entries:
        return render_template(
            "history_lookup.html",
            error=f"No saved history for {matric_no} yet — nothing to unlock with a PIN.",
        )

    trend = build_trend_chart(entries)
    return render_template("history_view.html", matric_no=matric_no, pin=pin, entries=entries, trend=trend)


@app.route("/history/delete", methods=["POST"])
def history_delete():
    matric_no = (request.form.get("matric_no") or "").strip()
    pin = request.form.get("pin") or ""
    try:
        delete_history(matric_no, pin)
        return render_template(
            "history_lookup.html",
            notice=f"History for {matric_no} was deleted.",
        )
    except WrongPin:
        return render_template("history_lookup.html", error="Incorrect PIN — nothing was deleted.")


@app.route("/about", methods=["GET"])
def about():
    metrics = {}
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    return render_template("about.html", metrics=metrics)


@app.route("/figures/<path:filename>", methods=["GET"])
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
