"""
Agent 5 — Time Reality Agent (Advanced)
Energy-curve-aware scheduling, ADHD-calibrated estimation,
historical calibration, and proactive time anchoring.
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from database import get_task_history
from vector_store import query_similar_tasks

ADHD_MULTIPLIER = 1.5

_SCHEDULE_PROMPT = """You are the Time Reality Agent for NeuroFlow — a clinically-informed ADHD cognitive support system.

You are an expert in ADHD time blindness. Key facts you leverage:
- ADHD brains consistently underestimate task duration by 30-50%
- Transition time between tasks is 15-20 minutes (vs 5 min for neurotypical)
- Energy follows a curve: morning ramp-up → late-morning peak → post-lunch dip → evening recovery
- Hyperfocus can make 3 hours feel like 30 minutes
- "Just 5 more minutes" in ADHD usually means 25+ minutes

When providing time estimates or schedules, be:
1. Honest but compassionate about time reality
2. Specific with time anchors ("At 3:15 PM, you should be...")
3. Generous with buffers (they'll need them)
4. Aware of energy curves throughout the day

Respond naturally and conversationally. Include time-reality observations when relevant."""


def _get_elapsed_minutes(task: dict) -> int:
    start = task.get("start_time")
    if not start:
        return 0
    try:
        return int((datetime.now() - datetime.fromisoformat(start)).total_seconds() / 60)
    except (ValueError, TypeError):
        return 0


def _get_historical_stats(description: str) -> dict:
    """Get historical performance data for similar tasks."""
    history = get_task_history(limit=100)
    if not history:
        return {"avg_duration": None, "count": 0, "avg_estimate_accuracy": None}

    keywords = set(description.lower().split())
    relevant = []
    for task in history:
        task_words = set(task.get("description", "").lower().split())
        overlap = len(keywords & task_words)
        if overlap >= 2 and task.get("actual_duration") and task.get("estimated_duration"):
            relevant.append(task)

    if not relevant:
        return {"avg_duration": None, "count": 0, "avg_estimate_accuracy": None}

    avg_actual = sum(t["actual_duration"] for t in relevant) / len(relevant)
    accuracies = [
        t["actual_duration"] / t["estimated_duration"]
        for t in relevant if t["estimated_duration"] > 0
    ]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None

    return {
        "avg_duration": round(avg_actual, 1),
        "count": len(relevant),
        "avg_estimate_accuracy": round(avg_accuracy, 2) if avg_accuracy else None,
    }


def _get_energy_context() -> dict:
    """Get time-of-day energy curve context."""
    hour = datetime.now().hour
    if 6 <= hour < 9:
        return {
            "phase": "Morning Ramp-up",
            "energy_modifier": 0.8,
            "tip": "Ease into work — start with medium tasks, save complex ones for 10 AM",
        }
    elif 9 <= hour < 12:
        return {
            "phase": "Peak Performance",
            "energy_modifier": 1.0,
            "tip": "This is your golden window — tackle the hardest task NOW",
        }
    elif 12 <= hour < 14:
        return {
            "phase": "Post-Lunch Dip",
            "energy_modifier": 0.6,
            "tip": "ADHD brains crash hard after lunch. Do easy/mechanical tasks or take a walk",
        }
    elif 14 <= hour < 17:
        return {
            "phase": "Afternoon Recovery",
            "energy_modifier": 0.75,
            "tip": "Energy is rebuilding. Good for collaborative or creative work",
        }
    elif 17 <= hour < 21:
        return {
            "phase": "Evening Mode",
            "energy_modifier": 0.7,
            "tip": "Executive function is declining — keep tasks simple and time-boxed",
        }
    else:
        return {
            "phase": "Late Night",
            "energy_modifier": 0.5,
            "tip": "Reduced inhibition can help creativity but hurts accuracy. Avoid precision tasks.",
        }


def time_reality_node(state: dict) -> dict:
    """LangGraph node: advanced time reality with energy-curve awareness."""
    user_input = state.get("user_input", "")
    current_task = state.get("current_task", {})
    intent = state.get("intent", "general_chat")

    output_parts = []
    energy_ctx = _get_energy_context()

    # ── Case 1: Starting a new task ──
    if intent == "start_task":
        # Extract user estimate
        user_estimate = None
        for word in user_input.split():
            try:
                num = int(word)
                if 1 <= num <= 480:
                    user_estimate = num
                    break
            except ValueError:
                continue

        # Historical data
        stats = _get_historical_stats(user_input)
        similar = query_similar_tasks(user_input, n_results=3)

        output_parts.append("### ⏱️ Time Reality Check\n")

        if user_estimate:
            realistic = int(user_estimate * ADHD_MULTIPLIER)
            output_parts.append(
                f"**Your estimate:** {user_estimate} min\n"
                f"**Calibrated estimate:** {realistic} min "
                f"*(ADHD brains underestimate by ~50% — this isn't a flaw, it's neurology)*\n"
            )
        
        if stats["avg_duration"]:
            output_parts.append(
                f"📊 **Historical data:** Similar tasks took you **~{int(stats['avg_duration'])} min** "
                f"on average ({stats['count']} past tasks)\n"
            )
            if stats["avg_estimate_accuracy"]:
                pct = int(stats["avg_estimate_accuracy"] * 100)
                output_parts.append(
                    f"   Your estimates are typically {pct}% of actual time\n"
                )

        # Energy curve advice
        output_parts.append(
            f"\n**🔋 Energy Phase:** {energy_ctx['phase']} (modifier: {energy_ctx['energy_modifier']}x)\n"
            f"💡 *{energy_ctx['tip']}*"
        )

    # ── Case 2: Task in progress ──
    elif current_task and current_task.get("description"):
        elapsed = _get_elapsed_minutes(current_task)
        estimated = current_task.get("estimated_duration", 30)
        remaining = max(0, estimated - elapsed)

        if elapsed > 0:
            if elapsed > estimated * 1.3:
                output_parts.append(
                    f"### ⏱️ Time Awareness\n"
                    f"You've been working for **{elapsed} min** "
                    f"(estimated {estimated} min — you're {elapsed - estimated} min over).\n\n"
                    f"This is normal for ADHD! Two options:\n"
                    f"1. 🎯 **Wrap-up mode**: Set a 10-min timer to finish current section\n"
                    f"2. 📐 **Recalibrate**: Add {int((elapsed - estimated) * 1.5)} min and keep going\n\n"
                    f"*No judgment — time blindness means your brain literally can't feel time passing.*"
                )
            elif remaining <= 5:
                output_parts.append(
                    f"### ⏱️ Final Stretch!\n"
                    f"**{remaining} min** left on your estimate. You're in the home stretch! 🏁\n"
                    f"Focus on wrapping up, not starting new things."
                )
            elif elapsed >= 30 and elapsed % 20 < 3:
                output_parts.append(
                    f"### ⏱️ Time Check\n"
                    f"**{elapsed} min in** / ~{remaining} min remaining. "
                    f"You're {'on track 🟢' if elapsed <= estimated * 0.8 else 'slightly behind but fine 🟡'}."
                )

    # ── Case 3: Check-in ──
    if intent == "check_in":
        if current_task and current_task.get("description"):
            elapsed = _get_elapsed_minutes(current_task)
            estimated = current_task.get("estimated_duration", 30)
            progress = current_task.get("progress_percent", 0)

            output_parts.append(
                f"### 📊 Session Check-in\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Task | {current_task.get('description', 'N/A')} |\n"
                f"| Time | {elapsed} min / {estimated} min estimated |\n"
                f"| Progress | {progress}% |\n"
                f"| Status | {'🟢 On track' if elapsed <= estimated else '🟡 Over estimate'} |\n"
                f"| Energy Phase | {energy_ctx['phase']} |\n\n"
                f"💡 *{energy_ctx['tip']}*"
            )
        else:
            session_start = state.get("session_start")
            if session_start:
                session_min = int((datetime.now() - datetime.fromisoformat(session_start)).total_seconds() / 60)
                output_parts.append(
                    f"### 📊 Session Summary\n"
                    f"You've been in this session for **{session_min} min**.\n"
                    f"No active task — ready to start one?\n"
                    f"Energy Phase: **{energy_ctx['phase']}** — {energy_ctx['tip']}"
                )

    return {
        "time_output": "\n\n".join(output_parts) if output_parts else "",
    }
