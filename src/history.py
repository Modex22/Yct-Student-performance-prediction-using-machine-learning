"""
Opt-in history: a student (or lecturer, on their behalf) can choose to save
a "Check My Score" result so it's possible to see a trend across terms
instead of one static snapshot. This is explicitly NOT automatic — nothing
here runs unless save_entry() is called from a request that ticked the
"save to history" box (see /quick-check/predict in app.py). Batch uploads
are never saved.

Access control: the first save for a matric number sets a PIN (stored only
as a salted hash); every later save, view, or delete for that matric number
must supply the same PIN. There's no recovery if it's forgotten — this is
a lightweight, self-service gate against a stranger browsing or polluting
someone else's saved history by guessing a matric number, not a real
authentication system. A real deployment holding actual student data
should sit behind proper institutional login instead.

Storage is a small SQLite file. On a host with an ephemeral filesystem
(e.g. Render's free tier without a persistent disk), this resets on every
redeploy — history isn't durable there without adding a paid persistent
disk or an external database. Documented in the README.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any

from src.config import HISTORY_DB_PATH

_PBKDF2_ITERATIONS = 200_000


class WrongPin(Exception):
    """The PIN supplied doesn't match the one set for this matric number."""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matric_no TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            exam_score REAL NOT NULL,
            test_score REAL NOT NULL,
            assignment_score REAL NOT NULL,
            practical_score REAL NOT NULL,
            predicted_score REAL NOT NULL,
            classification TEXT NOT NULL,
            pass_fail TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_matric ON history(matric_no)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history_access (
            matric_no TEXT PRIMARY KEY,
            pin_salt TEXT NOT NULL,
            pin_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _hash_pin(pin: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


def has_pin_set(matric_no: str) -> bool:
    """Whether this matric number already has history (and so an existing
    PIN) — lets the UI show "choose a PIN" vs "enter your PIN"."""
    matric_no = matric_no.strip()
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM history_access WHERE matric_no = ?", (matric_no,)
        ).fetchone()
    return row is not None


def _verify_or_register_pin(conn: sqlite3.Connection, matric_no: str, pin: str) -> None:
    """Raises WrongPin if a PIN is already set for this matric number and
    doesn't match. Registers `pin` as the PIN if none is set yet."""
    row = conn.execute(
        "SELECT pin_salt, pin_hash FROM history_access WHERE matric_no = ?", (matric_no,)
    ).fetchone()

    if row is None:
        salt = secrets.token_bytes(16)
        conn.execute(
            "INSERT INTO history_access (matric_no, pin_salt, pin_hash, created_at) VALUES (?, ?, ?, ?)",
            (matric_no, salt.hex(), _hash_pin(pin, salt), datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return

    salt = bytes.fromhex(row["pin_salt"])
    if not hmac.compare_digest(_hash_pin(pin, salt), row["pin_hash"]):
        raise WrongPin(f"Incorrect PIN for {matric_no}.")


def verify_pin(matric_no: str, pin: str) -> bool:
    """Read-only check: True if `pin` matches the matric number's PIN.
    False (never raises) if the matric number has no history/PIN yet, or
    the PIN is wrong — callers that need to distinguish "no history" from
    "wrong PIN" should check has_pin_set() first."""
    matric_no = matric_no.strip()
    with _connect() as conn:
        row = conn.execute(
            "SELECT pin_salt, pin_hash FROM history_access WHERE matric_no = ?", (matric_no,)
        ).fetchone()
    if row is None:
        return False
    salt = bytes.fromhex(row["pin_salt"])
    return hmac.compare_digest(_hash_pin(pin, salt), row["pin_hash"])


def save_entry(matric_no: str, pin: str, student: dict[str, Any], result: dict[str, Any]) -> None:
    """Record one saved check for a matric number. Called only when the
    person submitting explicitly opted in. Sets the PIN on first use for
    this matric number; raises WrongPin if a later save supplies the wrong
    one (so a stranger can't quietly append entries to someone else's
    history just by guessing their matric number)."""
    matric_no = matric_no.strip()
    if not matric_no:
        raise ValueError("matric_no is required to save to history")
    if not pin or len(pin) < 4:
        raise ValueError("A PIN of at least 4 characters is required to save to history")

    with _connect() as conn:
        _verify_or_register_pin(conn, matric_no, pin)
        conn.execute(
            """
            INSERT INTO history
                (matric_no, saved_at, exam_score, test_score, assignment_score,
                 practical_score, predicted_score, classification, pass_fail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                matric_no,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                student["exam_score"],
                student["test_score"],
                student["assignment_score"],
                student["practical_score"],
                result["predicted_score"],
                result["classification"],
                result["pass_fail"],
            ),
        )


def get_history(matric_no: str, pin: str) -> list[dict[str, Any]]:
    """All saved entries for a matric number, oldest first. Raises WrongPin
    if the PIN doesn't match an existing record; returns [] if there's no
    history (and so no PIN) for this matric number at all."""
    matric_no = matric_no.strip()
    if not has_pin_set(matric_no):
        return []
    if not verify_pin(matric_no, pin):
        raise WrongPin(f"Incorrect PIN for {matric_no}.")

    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE matric_no = ? ORDER BY saved_at ASC",
            (matric_no,),
        ).fetchall()
    return [dict(row) for row in rows]


def build_trend_chart(
    entries: list[dict[str, Any]],
    width: int = 480,
    height: int = 170,
    pad_x: int = 24,
    pad_y: int = 18,
) -> dict[str, Any] | None:
    """Pixel coordinates + SVG path strings for a simple line-and-area trend
    chart of predicted_score over saved entries, oldest to newest. Computed
    server-side (this app has no chart JS anywhere else either) so the
    template just drops the paths into an inline <svg>. None if there's
    nothing to plot."""
    if not entries:
        return None

    n = len(entries)
    plot_w = width - 2 * pad_x
    plot_h = height - 2 * pad_y

    def x_at(i: int) -> float:
        return pad_x if n == 1 else pad_x + (i / (n - 1)) * plot_w

    def y_at(score: float) -> float:
        return pad_y + plot_h - (max(0.0, min(100.0, score)) / 100) * plot_h

    points = [
        {
            "cx": round(x_at(i), 1),
            "cy": round(y_at(e["predicted_score"]), 1),
            "score": e["predicted_score"],
            "date": e["saved_at"][:10],
        }
        for i, e in enumerate(entries)
    ]

    baseline_y = pad_y + plot_h
    if n == 1:
        # A single point has no line to draw; still show the dot.
        line_path = ""
        area_path = ""
    else:
        line_path = "M " + " L ".join(f"{p['cx']},{p['cy']}" for p in points)
        area_path = (
            line_path
            + f" L {points[-1]['cx']},{baseline_y} L {points[0]['cx']},{baseline_y} Z"
        )

    return {
        "width": width,
        "height": height,
        "points": points,
        "line_path": line_path,
        "area_path": area_path,
        "baseline_y": baseline_y,
    }


def delete_history(matric_no: str, pin: str) -> int:
    """Remove every saved entry (and the PIN itself) for a matric number.
    Raises WrongPin if the PIN doesn't match; returns 0 if there's nothing
    to delete. Lets a student clear their own record and start over."""
    matric_no = matric_no.strip()
    if not has_pin_set(matric_no):
        return 0
    if not verify_pin(matric_no, pin):
        raise WrongPin(f"Incorrect PIN for {matric_no}.")

    with _connect() as conn:
        cursor = conn.execute("DELETE FROM history WHERE matric_no = ?", (matric_no,))
        conn.execute("DELETE FROM history_access WHERE matric_no = ?", (matric_no,))
        return cursor.rowcount
