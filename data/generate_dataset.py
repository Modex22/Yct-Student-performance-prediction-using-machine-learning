"""
Synthetic student performance dataset generator.

No public labeled dataset was supplied for this project, so this script
builds a realistic synthetic one, using only four inputs a department
already has on record for every student — no self-reported or hard-to-
collect data (attendance, lifestyle, family background, etc.):

    exam_score (/60), test_score (/10), assignment_score (/10),
    practical_score (/20)

Each component is scored on its own maximum (see COMPONENT_MAX in
src/config.py — a common Nigerian-polytechnic CA breakdown, not confirmed
against Yabatech's actual policy), and the four maximums already sum to
100 — so final_score is just their sum, no extra weighting needed; the
weighting is baked into each component's max mark. A small amount of noise
is added on top, representing real-world effects a pure sum wouldn't
capture exactly (moderation, rounding, marker variation), which is what
gives the model an actual job to do rather than just re-deriving addition.

Run directly to (re)write data/student_performance.csv:
    python data/generate_dataset.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Allow `python data/generate_dataset.py` to work directly (not just
# `python -m data.generate_dataset`) by ensuring the project root is on
# sys.path before importing the src package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import COMPONENT_MAX  # noqa: E402

RANDOM_SEED = 42
N_STUDENTS = 2000

# (mean as a fraction of the component's max, std as a fraction of max) —
# exams run harder and more spread out than coursework; assignments are
# the most forgiving. Same shape as before the components moved to their
# own maximums, just rescaled per component.
_DISTRIBUTION_FRACTIONS = {
    "exam_score": (0.58, 0.18),
    "test_score": (0.65, 0.15),
    "assignment_score": (0.72, 0.12),
    "practical_score": (0.68, 0.14),
}


def generate_dataset(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    components = {}
    for field, max_mark in COMPONENT_MAX.items():
        mean_frac, std_frac = _DISTRIBUTION_FRACTIONS[field]
        components[field] = np.clip(
            rng.normal(mean_frac * max_mark, std_frac * max_mark, n_students), 0, max_mark
        )

    noise = rng.normal(0, 2, n_students)
    final_score = sum(components.values()) + noise
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame({field: values.round(1) for field, values in components.items()})
    df["final_score"] = final_score

    df["performance_category"] = pd.cut(
        df["final_score"],
        bins=[-0.1, 49.9, 64.9, 79.9, 100],
        labels=["At Risk", "Average", "Good", "Excellent"],
    )
    df["pass_fail"] = np.where(df["final_score"] >= 50, "Pass", "Fail")

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    out_path = "data/student_performance.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Wrote {len(dataset)} rows to {out_path}")
    print(dataset.describe(include="all").transpose())
