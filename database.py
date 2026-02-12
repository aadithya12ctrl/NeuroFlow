"""
NeuroFlow Database Layer — SQLite
Creates tables and provides CRUD helpers for structured data.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "data", "neuroflow.db")


def _get_conn() -> sqlite3.Connection:
    """Return a connection with row-factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS interaction_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
            typing_speed REAL,
            message_length INTEGER,
            response_time_seconds REAL,
            typo_count  INTEGER DEFAULT 0,
            session_duration REAL,
            current_task_id TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pattern_events (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL DEFAULT (datetime('now')),
            pattern_type      TEXT NOT NULL,
            context           TEXT,
            intervention_used TEXT,
            success           INTEGER DEFAULT 0,
            user_response     TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_history (
            task_id           TEXT PRIMARY KEY,
            description       TEXT NOT NULL,
            estimated_duration INTEGER,
            actual_duration   INTEGER,
            completion_date   TEXT,
            energy_level_at_start INTEGER,
            interruptions_count INTEGER DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS time_blocks (
            block_id          TEXT PRIMARY KEY,
            start_time        TEXT NOT NULL,
            end_time          TEXT,
            task_id           TEXT,
            actual_productivity INTEGER DEFAULT 5
        )
    """)

    conn.commit()
    conn.close()


# ---- Interaction Metrics ----

def log_interaction(
    typing_speed: float = 0.0,
    message_length: int = 0,
    response_time: float = 0.0,
    typo_count: int = 0,
    session_duration: float = 0.0,
    task_id: Optional[str] = None,
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO interaction_metrics
           (typing_speed, message_length, response_time_seconds,
            typo_count, session_duration, current_task_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (typing_speed, message_length, response_time, typo_count,
         session_duration, task_id),
    )
    conn.commit()
    conn.close()


def get_recent_interactions(limit: int = 20) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM interaction_metrics ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Pattern Events ----

def log_pattern_event(
    pattern_type: str,
    context: str = "",
    intervention_used: str = "",
    success: bool = False,
    user_response: str = "",
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO pattern_events
           (pattern_type, context, intervention_used, success, user_response)
           VALUES (?, ?, ?, ?, ?)""",
        (pattern_type, context, intervention_used, int(success), user_response),
    )
    conn.commit()
    conn.close()


# ---- Task History ----

def save_task(
    task_id: str,
    description: str,
    estimated_duration: int = 30,
    actual_duration: Optional[int] = None,
    energy_level: int = 5,
    interruptions: int = 0,
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO task_history
           (task_id, description, estimated_duration, actual_duration,
            completion_date, energy_level_at_start, interruptions_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (task_id, description, estimated_duration, actual_duration,
         datetime.now().isoformat(), energy_level, interruptions),
    )
    conn.commit()
    conn.close()


def get_task_history(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM task_history ORDER BY completion_date DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Time Blocks ----

def save_time_block(
    block_id: str,
    start_time: str,
    end_time: Optional[str] = None,
    task_id: Optional[str] = None,
    productivity: int = 5,
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO time_blocks
           (block_id, start_time, end_time, task_id, actual_productivity)
           VALUES (?, ?, ?, ?, ?)""",
        (block_id, start_time, end_time, task_id, productivity),
    )
    conn.commit()
    conn.close()


# Auto-initialise when imported
init_db()
