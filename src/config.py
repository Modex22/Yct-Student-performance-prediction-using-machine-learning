"""Shared paths and column definitions used across the pipeline."""
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

REGRESSOR_PATH = os.path.join(MODELS_DIR, "score_regressor.joblib")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "pass_fail_classifier.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Opt-in "save to history" storage (see src/history.py). On a host with an
# ephemeral filesystem (e.g. Render free tier, no persistent disk) this
# resets on every redeploy — not durable there without a paid disk or an
# external database.
HISTORY_DB_PATH = os.path.join(BASE_DIR, "instance", "history.db")

# Just the four assessment components a department already records —
# nothing that has to be guessed, self-reported, or separately collected.
NUMERIC_FEATURES = [
    "exam_score",
    "test_score",
    "assignment_score",
    "practical_score",
]

CATEGORICAL_FEATURES: list[str] = []

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REGRESSION_TARGET = "final_score"
CLASSIFICATION_TARGET = "pass_fail"

# No categorical features currently — kept as an empty dict (rather than
# removed) so src/batch.py's validation loop stays generic if a
# categorical feature is added back later.
CATEGORY_VALUES: dict[str, list[str]] = {}

# Each component's own maximum mark, matching a common Nigerian-polytechnic
# continuous-assessment breakdown: Exam/60 + Test/10 + Assignment/10 +
# Practical/20 sums straight to 100 — no extra weighting needed, since the
# weighting is already built into each component's max. Not confirmed
# against Yabatech's actual CA policy; adjust here (and in
# data/generate_dataset.py) once known.
COMPONENT_MAX = {
    "exam_score": 60,
    "test_score": 10,
    "assignment_score": 10,
    "practical_score": 20,
}

CATEGORY_BINS = [-0.1, 49.9, 64.9, 79.9, 100]
CATEGORY_LABELS = ["At Risk", "Average", "Good", "Excellent"]

GRADE_BINS = [-0.1, 39.9, 49.9, 59.9, 69.9, 100]
GRADE_LABELS = ["F", "D", "C", "B", "A"]

# Nigerian-polytechnic-style (ND/HND) result classification. These cutoffs
# are an illustrative default modelled loosely on the common NBTE
# Distinction/Credit/Pass banding — they have NOT been confirmed against
# Yabatech's actual grading policy. Adjust them here if you have the
# official cutoffs.
CLASSIFICATION_BINS = [-0.1, 44.9, 49.9, 59.9, 69.9, 100]
CLASSIFICATION_LABELS = ["Fail", "Pass", "Lower Credit", "Upper Credit", "Distinction"]

# Non-predictive identity/roster columns a batch spreadsheet may include.
# They're carried through to the results table and dashboard for context
# but are never fed to the model. Each key maps to the accepted column
# name aliases a spreadsheet might use (matched case-insensitively).
IDENTITY_COLUMN_ALIASES = {
    "name": ["name", "student_name", "full_name", "student"],
    "matric_no": ["matric_no", "matric_number", "reg_no", "registration_number", "student_id"],
    "department": ["department", "dept"],
    "level": ["level", "class_level", "programme_level"],
}
IDENTITY_COLUMNS = list(IDENTITY_COLUMN_ALIASES.keys())

MAX_BATCH_ROWS = 1000
