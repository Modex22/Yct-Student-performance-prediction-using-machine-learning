# YABATECH Student Performance Predictor

A machine-learning system that predicts a student's likely final score, an
ND/HND-style classification, and pass/fail risk from four assessment
components a department already has on record for every student, each on
its own mark scheme (a common Nigerian-polytechnic CA breakdown that sums
straight to 100 — not confirmed against Yabatech's actual policy, see
`COMPONENT_MAX` in `src/config.py` to adjust):

- Exam score, out of 60
- Test / CA score, out of 10
- Assignment score, out of 10
- Practical score, out of 20

No lifestyle, demographic, or self-reported data (study habits, sleep,
family background, attendance, etc.) — just the numbers already sitting in
a results spreadsheet. Built for a class-roster workflow: upload that
spreadsheet, get a class report back.

**Not yet trained on real Yabatech data.** The model is trained on a
synthetic dataset (see below); the app is styled for Yabatech's context
(Department, Level, ND/HND classification) but the predictions themselves
are illustrative until it's retrained on real, anonymized records.

## How it works

1. **Dataset** (`data/generate_dataset.py`) — no labeled dataset was
   supplied, so a synthetic one (2,000 students, 4 features) is generated:
   each component score is sampled within its own max mark from a realistic
   distribution (exams run harder and more spread out than coursework;
   assignments are the most forgiving), and `final_score` is simply their
   sum (each component's max already encodes its weight — 60+10+10+20=100,
   so no extra weighting multiplier is needed) plus a little noise,
   representing real-world effects a pure sum wouldn't capture exactly
   (moderation, rounding, marker variation). Swap in a real dataset by
   replacing `data/student_performance.csv` with the same columns (see
   `src/config.py`).
2. **Preprocessing** (`src/data_preprocessing.py`) — the four scores are
   standardized via a `ColumnTransformer` that's part of the saved model
   pipeline (no separate scaler file to keep in sync). There are currently
   no categorical features, but the pipeline still has an (empty)
   categorical branch so one can be added back without restructuring
   anything.
3. **Training** (`src/train_model.py`) — trains and 5-fold cross-validates
   four regressors (Linear Regression, Ridge, Random Forest, Gradient
   Boosting) to predict the final score, keeps the best by test R², and
   separately trains a Random Forest classifier for pass/fail. Saves both
   pipelines, a `metrics.json`, and feature-importance / actual-vs-predicted
   plots. (Feature importance is read from `feature_importances_` for a
   tree ensemble or `|coefficient|` for a linear model — whichever type
   wins a given retrain.)
4. **Prediction** (`src/predict.py`) — loads the saved pipelines (training
   automatically on first use if they don't exist yet) and turns one
   student's four scores into a final score, grade, ND/HND classification,
   performance category, pass/fail call, rule-based recommendations, and a
   short "counsellor's note" (a one-line report-card-style remark naming
   the weakest component(s)).
5. **Batch upload** (`src/batch.py` + `/`) — the primary workflow: upload a
   `.csv`/`.xlsx` class roster, every row is validated and scored in one
   pass, and you get back a class dashboard (average score, pass rate,
   classification breakdown, score histogram, a ranked at-risk list) plus a
   full per-student results table. A scanning animation plays while the
   file is processed. `/batch/template` downloads a ready-to-fill CSV.
6. **Check my score** (`/quick-check`) — the same prediction for one
   student filling in their own four scores, instead of a roster. Shows a
   score breakdown (each component's raw contribution vs. the model's
   learned adjustment) and a confidence range (predicted score ± the
   model's test MAE, so a single decimal doesn't read as more precise
   than it is). Optionally, ticking "save to history" (with a matric
   number **and a PIN**) stores that one result — see below; nothing is
   saved by default.
7. **History** (`/history`, `src/history.py`) — opt-in only: look up a
   matric number *with its PIN* to see every result explicitly saved for
   it, plus a trend chart of predicted score over time. The PIN is set on
   the first save for a matric number and required for every later save,
   view, or delete against it — a stranger who finds/guesses a matric
   number alone can't read or add to someone else's saved history. It's a
   lightweight, self-service gate (a salted-hash PIN check, no accounts),
   not real authentication — a deployment handling actual student data
   should sit behind proper institutional login instead. Batch uploads are
   never saved here.
8. **Model info** (`/about`) — how each candidate model performed, and the
   caveats on the ND/HND classification cutoffs and the synthetic training
   data.

Current results on the synthetic dataset: best regressor is Linear
Regression (test R² ≈ 0.97, MAE ≈ 1.7 points on a 0-100 scale — very high
because `final_score` really is the sum of the four inputs by
construction; the model has to discover that on its own from noisy
examples, though — it's never given the formula. Its learned per-point
coefficients land within ~5% of 1.0 for every component, confirming it
recovered the actual relationship rather than memorizing anything); the
pass/fail classifier reaches ≈ 96% accuracy. Re-run training to regenerate
these numbers on real data — see below.

## Design

A dark, green-gradient sidebar layout (all in `static/style.css`, tokens at
the top) — a deliberate departure from the earlier light "academic ledger"
look, matching a dashboard reference the app was asked to be styled after.
Status colors stay semantic regardless of the brand accent: red for
Fail/At Risk, amber for Pass/Average, green for Good/Distinction — the
brand being green never means "everything is green." The two matplotlib
plots on `/about` are themed to match (`_apply_dark_theme()` in
`src/train_model.py`) rather than sitting as a white rectangle on a black
page. No JS charting library anywhere — the trend chart and the batch
dashboard's bars are all plain inline SVG / CSS widths computed
server-side, consistent with how the rest of the app already worked.

## Project structure

```
├── app.py                     # Flask web app (batch upload is "/")
├── data/
│   ├── generate_dataset.py    # synthetic dataset generator
│   └── student_performance.csv
├── src/
│   ├── config.py              # paths, feature/target columns, classification bins
│   ├── data_preprocessing.py  # ColumnTransformer + data loading
│   ├── train_model.py         # trains, compares, saves models + plots
│   ├── predict.py             # loads models, predicts one student (+ breakdown, MAE)
│   ├── batch.py                # parses a roster spreadsheet, runs batch predictions
│   ├── history.py             # opt-in save/lookup/delete + trend-chart math (SQLite)
│   ├── export_web_model.py    # exports the trained model to JSON for a client-side demo
│   └── export_web_charts.py   # exports chart data for that demo
├── models/                    # saved pipelines + metrics.json (generated)
├── reports/figures/           # feature importance & actual-vs-predicted plots (dark theme)
├── instance/                  # history.db (generated, gitignored — see Deploying)
├── templates/, static/        # Flask views: dark/green sidebar layout
└── tests/                     # pytest sanity checks
```

## Setup

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `py` is unavailable, install Python 3.11 or later and enable the Python
launcher during installation.

For a local deployment, configure the account credentials before starting
the app. PowerShell example (replace the sample values):

```powershell
$env:SECRET_KEY = "replace-with-a-long-random-secret"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "replace-with-a-strong-admin-password"
$env:LECTURER_USERNAME = "lecturer"
$env:LECTURER_PASSWORD = "replace-with-a-strong-lecturer-password"
```

Do not commit credentials or real student spreadsheets to source control.
The guest demo does not require an account.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the models

```bash
python data/generate_dataset.py   # (re)generate the dataset
python -m src.train_model         # train, compare, save models + plots
```

This writes `models/score_regressor.joblib`, `models/pass_fail_classifier.joblib`
and `models/metrics.json`. If you skip this step, the web app and CLI train
automatically the first time a prediction is requested.

## Run the web app

```bash
python app.py
```

Then open http://localhost:5000 — the batch upload page is the home page.
Click "Try an example roster" to see the full flow without your own file,
or download the CSV template and fill in a real class list. `/quick-check`
lets one student check their own score, and `/about` has model metrics and
charts.

### Student guest demo

Open `http://localhost:5000/demo` or choose **Try guest demo** on the home
page. No student account is needed. Enter made-up values to try the exam
estimate; the form does not query student accounts or saved scores and does
not save submissions. The page labels output as a demo estimate. Exam
predictions require the lecturer to have trained the exam model. A
second-semester estimate is available only after the administrator has
trained the separate paired-semester model; otherwise the page explains
that the model is unavailable. Admin and lecturer dashboards, uploads, and
training remain behind their existing role-protected logins.

### Pass-rate reference

Open `/benchmarks` from the public menu to compare estimated pass rates by
YABATECH course with public pass rates grouped by reported study hours.
These are separate aggregate summaries, not joined student records and not
a combined prediction model. The YABATECH workbook values are estimated
demonstration scores. The public UCI dataset has 145 records and grouped
weekly study-hour responses; small groups make its rates descriptive only.

## Deploying it publicly

**Not Vercel** — this app's dependencies (numpy + pandas + scipy + scikit-learn)
total ~283 MB uncompressed, over Vercel's ~250 MB serverless function limit,
and batch upload needs a real running Python process (pandas parsing an
uploaded file), not a stateless function. Vercel *is* a great fit for the
"Grade Forecast" client-side demo (`src/export_web_model.py`) since that's
pure static HTML/JS with no backend at all.

For the full app (including batch upload), any host that runs a persistent
Python process works — this repo is ready for **Render**
(`render.yaml` + `Procfile`, both already here):

1. Push this repo to your own GitHub account (or use this one directly if
   you have push access).
2. On [render.com](https://render.com): New → Blueprint → connect the repo.
   It reads `render.yaml` automatically — free tier, no config needed.
3. Or manually: New → Web Service → connect the repo → Render detects
   Python, build command `pip install -r requirements.txt`, start command
   `gunicorn app:app` (already set in `render.yaml`).

Railway, Fly.io, PythonAnywhere, or a plain VM all work the same way — the
only requirement is running `gunicorn app:app` (or equivalent) instead of
`python app.py`'s development server. Verified locally before recommending
this: `gunicorn app:app` serves every route (including a full predict
request) identically to the dev server.

**History persistence**: `instance/history.db` (SQLite) is created on first
use and is a normal file on local disk — fine for a VM or a Render instance
with a persistent disk attached. Render's **free** tier has no persistent
disk, so that file resets on every redeploy or restart there; the
save-to-history feature would appear to work but silently lose data over
time. Either attach a paid persistent disk, or swap `src/history.py` for a
real external database (Postgres, etc.) before relying on it in production.

### Roster spreadsheet format

A `.csv` or `.xlsx` with one row per student. Required columns are the 4
feature columns in `src/config.py:FEATURE_COLUMNS` — `exam_score` (0-60),
`test_score` (0-10), `assignment_score` (0-10), `practical_score` (0-20).
Optional identity columns — `name`, `matric_no`, `department`, `level`
(aliases like `student_name` or `dept` are also recognized) — are carried
through to the report for context but never fed to the model. Non-numeric
cells are flagged as warnings and that row is skipped rather than crashing
the whole upload; a value outside its column's expected range (e.g. an
exam score of 95 where the max is 60 — the likely mistake being entering
it out of 100 instead) is flagged but still processed, in case it's
legitimate.

## Static client-side demo (no server)

`src/export_web_model.py` exports the fitted regressor + classifier
pipelines to a JSON file a plain-JS predictor can walk with no Python/Flask
backend required, and `src/export_web_charts.py` exports the
feature-importance and actual-vs-predicted data used by the demo's charts.
The regressor's shape depends on which model type won training — a tree
ensemble is exported as its trees (feature/threshold/children/value per
node), a linear model as its coefficients + intercept — so the exporter
(and the JS that reads its output) branches on `payload.regressor.type`
rather than assuming one shape:

```bash
python -m src.export_web_model web_demo_model.json
python -m src.export_web_charts web_demo_charts.json
```

Used to build the "Grade Forecast" artifact demo (single-student only —
the batch upload workflow is Flask-only for now).

## Predict from the command line

```bash
python -m src.predict
```

Edit the `example_student` dict at the bottom of `src/predict.py`, or import
`predict_performance()` from your own script — it takes a dict with the keys
listed in `src/config.py:FEATURE_COLUMNS` (`exam_score`, `test_score`,
`assignment_score`, `practical_score`) and returns the predicted score,
grade, ND/HND classification, category, pass/fail call, recommendations,
and a counsellor's note.

## Tests

```bash
pytest
```

## Notes on adapting this to real data

Replace `data/student_performance.csv` with real records that (a) keep the
same column names as `FEATURE_COLUMNS` in `src/config.py` (or update that
list, and `COMPONENT_MAX`, if the real assessment structure or mark scheme
differs) and (b) include a `final_score` (0-100) target and a `pass_fail`
(`Pass`/`Fail`) target, then re-run `python -m src.train_model`. Everything
downstream (preprocessing, training, the web app) works unchanged. The
`CLASSIFICATION_BINS`/`CLASSIFICATION_LABELS` cutoffs in `src/config.py` are
an illustrative default — update them to Yabatech's actual ND/HND grading
policy once known.


## Student login portal update (November 2026)

This version adds student registration/login, a student dashboard, a first-semester score-entry form for the 16 course codes in the supplied workbook, saved submission history, and an administrator dashboard. Passwords are stored as Werkzeug password hashes in `instance/accounts.db`; student score submissions are stored per account. The guest-demo ZIP excludes the supplied workbook and any local account database; keep real or student-level source files out of shared archives.

### Important machine-learning limitation

The supplied workbook identifies itself as an **estimated demonstration dataset** and contains only one semester. It does not contain paired HND1 first- and second-semester outcomes. Therefore, this application deliberately does **not** claim to make a trained second-semester prediction. The student portal currently calculates a descriptive first-semester average and course-level performance bands. This is not a machine-learning forecast. To train and validate the intended second-semester predictor, obtain anonymized, paired records for the same students: first-semester course scores as features and second-semester course scores or semester average as target labels. Do not treat the workbook's estimated values as verified institutional records.

### Local setup and administrator account

Set a long random `SECRET_KEY`, plus `ADMIN_PASSWORD` before starting the app. Optionally set `ADMIN_USERNAME` (defaults to `admin`). On first start, the app creates the administrator account if `ADMIN_PASSWORD` is set. Students can register through `/register`. Never deploy with the development fallback secret or a blank admin password. Example (Linux/macOS):

```bash
cp .env.example .env  # set these values in your shell or hosting dashboard
export SECRET_KEY='replace-with-a-long-random-secret'
export ADMIN_USERNAME='admin'
export ADMIN_PASSWORD='choose-a-strong-admin-password'
python app.py
```

The admin login is at `/login`; the admin dashboard is `/admin`. Student registration is at `/register`. The existing batch model remains a demonstration workflow and should not be described as a second-semester predictor.

### Deployment

This remains a Flask/Python server application. Deploy it to a persistent Python host such as Render or Railway rather than trying to force the full scikit-learn/pandas application into a Vercel serverless function. Configure `SECRET_KEY`, `ADMIN_USERNAME`, and `ADMIN_PASSWORD` as deployment environment variables. SQLite is suitable for local demonstration only; for multiple users in production, use managed PostgreSQL and persistent storage.

## YABATECH student portal refresh (November 2026)

This update adds a responsive dark navy/green student portal experience while retaining the Flask application and existing batch-analysis tools.

### Portal features
- Public landing page with student login and registration entry points.
- Student registration/login using matriculation number and password hashes.
- Student dashboard with saved submission history and course-score visual summaries.
- First-semester course score entry and descriptive performance analysis.
- Admin overview with student-account listing and submission statistics.
- Admin batch-analysis entry point.
- Responsive layouts for desktop and mobile.

### Dataset and ML limitations
The supplied workbook `data/yabatech_student_courses_grades_gpa_verified.xlsx` is included for inspection and development. Its Read Me sheet describes the values as estimated demonstration data, and it contains first-semester-style course records rather than paired first- and second-semester outcomes. Therefore, it is **not valid training data for a supervised HND1 first-to-second-semester prediction model**.

The portal currently performs descriptive analysis of submitted first-semester course scores. It must not display invented second-semester predictions or claim that the demonstration model is a validated semester-transition model. To enable genuine second-semester predictions, obtain consented/anonymized historical records with matching student identifiers, course codes, first-semester scores and second-semester target scores. Evaluate using a student-level holdout split to avoid leakage between the same student's course records.

### Admin setup
Set a strong `SECRET_KEY`, `ADMIN_USERNAME`, and `ADMIN_PASSWORD` in the deployment environment before production use. The SQLite database is created under `instance/` at startup. For multi-instance/production deployment, use a persistent database (e.g. PostgreSQL) rather than ephemeral local storage.

## Super Admin usage monitoring (portal upgrade)

The protected `/admin` dashboard now includes usage analytics backed by two additional SQLite tables:

- `activity_logs`: successful logins, failed login attempts, registrations, logouts, score submissions and quick-check activity.
- `prediction_logs`: prediction/analysis request type, status and optional duration.

The dashboard reports registered students, unique active users today, successful logins today, prediction/analysis counts, failed prediction counts, a seven-day prediction activity chart, recent events and the most active students. Metrics are refreshed when an administrator reloads the dashboard; this is not a streaming live feed.

### Railway persistence

The account database is stored at `instance/accounts.db`. Attach a Railway Volume mounted at `/app/instance` to preserve accounts and usage logs across redeployments. Keep `SECRET_KEY`, `ADMIN_USERNAME` and `ADMIN_PASSWORD` in Railway Variables, not in source control. The initial admin account is created from `ADMIN_USERNAME` and `ADMIN_PASSWORD` when the database is initialized; set these before the first deployment. If the database already exists, changing the variables does not automatically reset an existing admin password.

Usage logs intentionally store event names, user references and limited metadata, not passwords or submitted score values. Inform students that account activity is monitored and restrict dashboard access to trusted administrators.

## Admin-only paired-semester model training (new)

The administrator dashboard now links to **Model training** at `/admin/training`. Only an authenticated account whose database role is `admin` can open this page or submit a training file. Configure `ADMIN_USERNAME` and a strong `ADMIN_PASSWORD` in your local `.env` or Railway Variables before the first database initialization. The password is stored as a Werkzeug password hash, not plaintext. Student registration always creates `student` accounts; it cannot create administrators. If the database already contains the admin account, changing Railway variables alone does not reset its password.

The training upload accepts CSV or XLSX with exactly these required fields (additional columns are ignored):

- `student_id` — stable pseudonymous student identifier
- `course` — course code, e.g. `SWD 321`
- `first_semester_score` — first-semester score from 0 to 100
- `second_semester_score` — actual second-semester score from 0 to 100 (the target)

Download the example from the admin page. Each student/course pair must be unique, values cannot be blank, and at least 10 distinct students are required. The evaluation holds out whole students (rather than random rows) to reduce leakage. It compares three regressors using MAE, RMSE and R², then saves a separate artifact at `models/semester_transition_model.joblib` and metrics at `models/semester_transition_metrics.json`.

**Important:** this training page creates a separate first-to-second-semester model. It does not activate second-semester predictions in the student interface yet. That integration should only be done after validating real paired YABATECH records. The included estimated/demo workbook is not suitable for this training task. Do not claim its demonstration metrics as real model performance.


## Student, lecturer and mobile portal updates

### Role access
- Students register and sign in at `/login`.
- Lecturers sign in separately at `/lecturer/login`. Their account is created on startup when `LECTURER_USERNAME` and `LECTURER_PASSWORD` are configured.
- Administrators sign in with `ADMIN_USERNAME` and `ADMIN_PASSWORD`; administrator-only training and analytics routes remain protected by role checks.

Set these Railway variables (use strong private values):
```env
SECRET_KEY=your-long-random-secret
ADMIN_USERNAME=modexadmin
ADMIN_PASSWORD=your-strong-admin-password
LECTURER_USERNAME=lecturer
LECTURER_PASSWORD=your-strong-lecturer-password
```

### Lecturer assessment spreadsheet
The lecturer dashboard accepts CSV/XLSX files with one row per student per course. Required columns are:
`student_id` (or `matric_no`), `course`, `study_time_hours`, `test_score`, `practical_score`, and `exam_score`. `assignment_score` is optional and is shown to the matching student. The uploader stores the course results and trains the current-exam estimator. At least 10 distinct students with complete, real historical data are required.

The exam estimator uses only `study_time_hours`, `test_score`, and `practical_score` as requested; assignment score is collected for the student course profile but is not a model input. Students can see lecturer-published results when the spreadsheet matric number matches their account matric number.

### Student predictors
The student course predictor collects course, weekly study hours, assignment, practical and test scores. It can estimate the current exam score after a lecturer has trained the exam model. It also offers a second-semester estimate only after an administrator has uploaded paired historical first- and second-semester data through **Admin → Model Training**. The second-semester model uses course and first-semester course score.

The displayed course estimate is an indicative simple average of the entered assessment components and predicted exam score, not an official weighted course total. Replace this with the department's official assessment weighting if confirmed. Predictions are estimates, not guaranteed grades.

### Mobile layout
The sidebar navigation collapses into a dropdown menu on small screens. Student, lecturer and admin pages use responsive grids and scrollable tables.
