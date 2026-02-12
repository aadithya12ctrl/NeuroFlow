"""
NeuroFlow Utilities — Interaction Metrics
Helpers for computing typing speed, detecting trends, and break suggestions.
"""

from datetime import datetime
from typing import Optional


def compute_typing_speed(text: str, elapsed_seconds: float) -> float:
    """Return characters per second. Returns 0 if elapsed is ≤ 0."""
    if elapsed_seconds <= 0:
        return 0.0
    return len(text) / elapsed_seconds


def detect_trend(values: list[float], window: int = 5) -> str:
    """
    Return 'increasing', 'stable', or 'decreasing' based on the
    last `window` values.
    """
    if len(values) < 2:
        return "stable"
    recent = values[-window:]
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff > 0.1:
        return "increasing"
    elif avg_diff < -0.1:
        return "decreasing"
    return "stable"


def should_suggest_break(
    session_start: Optional[str],
    last_break: Optional[str],
    threshold_minutes: int = 90,
) -> tuple[bool, int]:
    """
    Returns (should_break, minutes_since_last_rest).
    Uses `last_break` if available, otherwise `session_start`.
    """
    now = datetime.now()
    if last_break:
        ref = datetime.fromisoformat(last_break)
    elif session_start:
        ref = datetime.fromisoformat(session_start)
    else:
        return False, 0

    minutes = int((now - ref).total_seconds() / 60)
    return minutes >= threshold_minutes, minutes


def compute_message_stats(messages: list[str]) -> dict:
    """Return basic stats about a list of user messages."""
    if not messages:
        return {"avg_length": 0, "min_length": 0, "max_length": 0, "count": 0}
    lengths = [len(m) for m in messages]
    return {
        "avg_length": sum(lengths) // len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "count": len(lengths),
    }
