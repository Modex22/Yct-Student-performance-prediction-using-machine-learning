"""Aggregated, clearly separated pass-rate references for the public demo page.

These are summary counts only. No student-level records are included here.
The YABATECH values come from the estimated demonstration workbook; the
study-time values come from UCI's Higher Education Students Performance
Evaluation dataset. They must not be joined or treated as one training set.
"""

YABATECH_SOURCE = "Estimated demonstration workbook (100 records per course)"
YABATECH_PASS_RULE = "Estimated score of 40 or above"
YABATECH_COURSE_RATES = [
    {"course": "COM 301", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 311", "students": 100, "passed": 96, "rate": 96.0},
    {"course": "COM 314", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 317", "students": 100, "passed": 99, "rate": 99.0},
    {"course": "COM 401", "students": 100, "passed": 99, "rate": 99.0},
    {"course": "COM 411", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 412", "students": 100, "passed": 99, "rate": 99.0},
    {"course": "COM 413", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 414", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 415", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 416", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "COM 417", "students": 100, "passed": 99, "rate": 99.0},
    {"course": "COM 418", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "GNS 301", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "STA 301", "students": 100, "passed": 100, "rate": 100.0},
    {"course": "STA 314", "students": 100, "passed": 100, "rate": 100.0},
]

PUBLIC_SOURCE = "UCI Higher Education Students Performance Evaluation (145 records)"
PUBLIC_PASS_RULE = "Grade 0 = fail; grades 1–7 = pass"
PUBLIC_STUDY_HOUR_RATES = [
    {"band": "No study time", "records": 29, "passed": 28, "rate": 96.6},
    {"band": "Under 5 hours/week", "records": 74, "passed": 70, "rate": 94.6},
    {"band": "6–10 hours/week", "records": 30, "passed": 28, "rate": 93.3},
    {"band": "11–20 hours/week", "records": 8, "passed": 8, "rate": 100.0},
    {"band": "Over 20 hours/week", "records": 4, "passed": 3, "rate": 75.0},
]
