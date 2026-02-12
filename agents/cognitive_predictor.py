"""
Agent 2 — Cognitive State Predictor
Advanced multi-factor cognitive state analysis with weighted scoring,
exponential moving averages, and neuroscience-backed intervention strategies.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from state import CognitiveState, CrashPrediction, InteractionMetrics
from utils.metrics import detect_trend, should_suggest_break


# ── Weighted Scoring Configuration (from README spec) ──
# Crash score = weighted sum of 6 behavioral factors
WEIGHTS = {
    "typing_speed_decline": 0.25,    # Strongest fatigue signal
    "message_length_trend": 0.20,    # Working memory capacity
    "response_time_trend": 0.20,     # Processing speed
    "break_overdue": 0.15,           # Proxy for typo/motor control degradation
    "topic_drift": 0.10,             # Executive function decline
    "session_duration": 0.10,        # Sigmoid fatigue curve
}


def _exponential_moving_avg(values: list[float], alpha: float = 0.3) -> float:
    """Compute exponential weighted moving average — recent values weigh more."""
    if not values:
        return 0.0
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _compute_crash_score(metrics: InteractionMetrics, session_minutes: float) -> dict:
    """
    Multi-factor weighted crash likelihood scoring.
    Returns dict with overall score and per-factor breakdown.
    """
    factors = {}

    # Factor 1: Typing speed decline from baseline
    if metrics.typing_speed_baseline > 0 and metrics.current_typing_speed > 0:
        speed_ratio = metrics.current_typing_speed / metrics.typing_speed_baseline
        # 1.0 = no decline, 0.5 = 50% decline
        factors["typing_speed_decline"] = max(0, 1.0 - speed_ratio)
    else:
        factors["typing_speed_decline"] = 0.0

    # Factor 2: Message length trend (EMA-based)
    if len(metrics.message_lengths) >= 3:
        lengths = [float(l) for l in metrics.message_lengths]
        ema_recent = _exponential_moving_avg(lengths[-5:])
        ema_overall = sum(lengths) / len(lengths)
        if ema_overall > 0:
            decline = max(0, 1.0 - (ema_recent / ema_overall))
            factors["message_length_trend"] = min(decline * 2, 1.0)  # amplify signal
        else:
            factors["message_length_trend"] = 0.0
    else:
        factors["message_length_trend"] = 0.0

    # Factor 3: Response time trend (increasing = fatigue signal)
    if len(metrics.response_times) >= 3:
        times = metrics.response_times[-8:]
        ema_recent = _exponential_moving_avg(times[-3:])
        ema_early = _exponential_moving_avg(times[:3])
        if ema_early > 0:
            increase = max(0, (ema_recent - ema_early) / ema_early)
            factors["response_time_trend"] = min(increase, 1.0)
        else:
            factors["response_time_trend"] = 0.0
    else:
        factors["response_time_trend"] = 0.0

    # Factor 4: Session duration fatigue curve (sigmoid around 90 min)
    if session_minutes > 0:
        # Sigmoid: rises sharply around 90 minutes
        factors["session_duration"] = 1.0 / (1.0 + math.exp(-(session_minutes - 90) / 20))
    else:
        factors["session_duration"] = 0.0

    # Factor 5: Break overdue
    needs_break, mins_since = should_suggest_break(None, metrics.last_break, threshold_minutes=45)
    if needs_break:
        factors["break_overdue"] = min(1.0, (mins_since - 45) / 45)
    else:
        factors["break_overdue"] = 0.0

    # Factor 6: Topic drift (proxy for executive function decline)
    # Short, unfocused messages or rapid changes suggest wandering attention
    if len(metrics.message_lengths) >= 4:
        recent_lens = metrics.message_lengths[-4:]
        variance = sum((l - sum(recent_lens)/len(recent_lens))**2 for l in recent_lens) / len(recent_lens)
        # High variance in message lengths suggests erratic engagement
        factors["topic_drift"] = min(1.0, variance / 5000)
    else:
        factors["topic_drift"] = 0.0

    # Weighted sum
    overall = sum(WEIGHTS[k] * factors[k] for k in WEIGHTS)

    return {"overall": round(min(overall, 1.0), 3), "factors": factors}


def _determine_focus(metrics: InteractionMetrics, crash_score: dict) -> str:
    """Determine focus level from metrics and crash analysis."""
    density = crash_score["factors"].get("topic_drift", 0)
    msg_trend = detect_trend([float(l) for l in metrics.message_lengths[-8:]])

    # Hyperfocus: low topic drift, consistent message lengths, no fatigue
    if (density < 0.1
            and crash_score["overall"] < 0.3
            and msg_trend != "decreasing"
            and len(metrics.message_lengths) > 5):
        return "hyperfocus"

    # High: moderate engagement, low crash risk
    if crash_score["overall"] < 0.25 and metrics.avg_message_length > 40:
        return "high"

    # Low: high crash risk or declining metrics
    if crash_score["overall"] > 0.5 or msg_trend == "decreasing":
        return "low"

    return "medium"


def _generate_intervention(focus: str, crash: dict, metrics: InteractionMetrics) -> str:
    """Generate contextual, neuroscience-backed intervention messages."""
    score = crash["overall"]
    factors = crash["factors"]

    if score >= 0.7:
        # Critical — multiple factors converging
        high_factors = [k for k, v in factors.items() if v > 0.5]
        evidence = ", ".join(k.replace("_", " ") for k in high_factors[:2])
        return (
            f"⚠️ **Cognitive Overload Warning**\n\n"
            f"Multiple fatigue indicators are converging ({evidence}). "
            f"Neuroscience research shows that pushing through this state actually *reduces* "
            f"total output compared to taking a break.\n\n"
            f"**Recommended Reset Protocol:**\n"
            f"1. 🧍 Stand up and stretch for 60 seconds\n"
            f"2. 💧 Drink water (dehydration amplifies ADHD symptoms)\n"
            f"3. 👀 Look at something 20+ feet away for 20 seconds\n"
            f"4. 🌬️ Take 3 deep breaths (activates parasympathetic nervous system)\n"
            f"5. ⏱️ Set a 5-minute timer, then come back refreshed"
        )

    if score >= 0.45:
        return (
            f"💡 **Gentle Check-in**\n\n"
            f"Your cognitive metrics are showing early fatigue signs "
            f"(crash likelihood: {int(score * 100)}%). "
            f"This is the *optimal* time for a micro-break — catching it "
            f"early means you can sustain focus much longer.\n\n"
            f"Quick options:\n"
            f"- 🚶 2-minute walk (resets default mode network)\n"
            f"- 🎵 Listen to one song (dopamine boost)\n"
            f"- ✋ Hand stretches (reduces screen fatigue)"
        )

    if focus == "hyperfocus":
        needs_break, mins = should_suggest_break(None, metrics.last_break, 60)
        if needs_break:
            return (
                f"🟣 **Hyperfocus Detected** — {mins}min deep\n\n"
                f"You're in a powerful flow state. I don't want to break it, "
                f"but your brain needs fuel to sustain this. Quick deal: "
                f"finish your current thought, grab water, and come right back. "
                f"Your flow state will survive a 2-minute pause."
            )

    return ""


def cognitive_predictor_node(state: dict) -> dict:
    """LangGraph node: advanced multi-factor cognitive state analysis."""
    raw_metrics = state.get("interaction_metrics", {})
    metrics = InteractionMetrics(**raw_metrics) if raw_metrics else InteractionMetrics()

    raw_cognitive = state.get("cognitive_state", {})
    prev = CognitiveState(**raw_cognitive) if raw_cognitive else CognitiveState()

    # Calculate session duration
    session_start = state.get("session_start")
    session_minutes = 0
    if session_start:
        try:
            session_minutes = (datetime.now() - datetime.fromisoformat(session_start)).total_seconds() / 60
        except (ValueError, TypeError):
            pass

    # Multi-factor crash scoring
    crash_score = _compute_crash_score(metrics, session_minutes)
    focus = _determine_focus(metrics, crash_score)
    intervention = _generate_intervention(focus, crash_score, metrics)

    # Dopamine model: decays naturally, boosted by task completion and milestones
    dopamine = max(0, min(100, prev.dopamine_balance - 1))

    # Energy model: influenced by crash score
    if crash_score["overall"] > 0.6:
        energy = max(1, prev.energy_level - 2)
    elif crash_score["overall"] > 0.3:
        energy = max(2, prev.energy_level - 1)
    else:
        energy = min(10, prev.energy_level)  # stable or slight recovery

    crash_minutes = max(5, int(60 * (1 - crash_score["overall"])))

    new_cognitive = CognitiveState(
        focus_level=focus,
        energy_level=energy,
        dopamine_balance=dopamine,
        crash_prediction=CrashPrediction(
            likelihood=round(crash_score["overall"], 2),
            estimated_minutes=crash_minutes,
        ),
    )

    return {
        "cognitive_state": new_cognitive.model_dump(),
        "cognitive_output": intervention,
    }
