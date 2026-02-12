"""
NeuroFlow — Dopamine Economy Manager (Agent 7)
Tracks motivation as a depletable daily budget (0-100).
Applies transaction scoring, variable-ratio reward scheduling,
and budget-based recommendations.
"""

from __future__ import annotations

import random
import traceback
from datetime import datetime

from state import NeuroFlowState, DopamineEconomy, DopamineTransaction


# ---------------------------------------------------------------------------
# Transaction scoring table (from README spec)
# ---------------------------------------------------------------------------

TRANSACTIONS = {
    # Gains
    "task_started":           +15,
    "task_completed":         +10,
    "small_milestone":        +5,
    "took_break_before_crash": +8,
    "pattern_interrupted":    +12,
    "high_five":              +3,

    # Costs
    "doom_scrolled_15min":    -20,
    "missed_planned_task":    -15,
    "context_switch":         -10,
    "self_criticism":         -8,
}


def _generate_variable_schedule(total_duration: int = 60) -> list[int]:
    """Generate variable-ratio reward intervals (slot-machine psychology).
    NOT every 25 min — unpredictable = more engaging for ADHD brains."""
    intervals = []
    t = 0
    min_gap, max_gap = 5, 30
    while t < total_duration:
        remaining = total_duration - t
        if remaining < min_gap:
            break  # Not enough time left for another interval
        upper = min(max_gap, remaining)
        gap = random.randint(min_gap, upper)
        t += gap
        if t <= total_duration:
            intervals.append(t)
    return intervals


def _compute_forecast(balance: int) -> str:
    """Predict energy outlook based on current balance."""
    if balance < 30:
        return "⛈️ Low energy — plan easy wins to rebuild motivation"
    elif balance < 50:
        return "🌥️ Moderate energy — mix easy tasks with one moderate challenge"
    elif balance < 70:
        return "⛅ Good energy — you can tackle something meaningful"
    else:
        return "☀️ High energy — perfect time for that hard task you've been avoiding!"


def _get_recommendation(balance: int, task_type: str = "general") -> str:
    """Budget-based recommendation."""
    if balance < 30:
        suggestions = [
            "Do something EASY and rewarding to rebuild motivation fuel",
            "Try a quick win — organise one folder, reply to one email",
            "Take a proper break with a specific reward (snack, music, walk)",
        ]
        return random.choice(suggestions)
    elif balance < 50:
        return "You have moderate motivation. Try a 15-min sprint on something manageable."
    elif balance > 70:
        suggestions = [
            "High energy! Perfect time for that hard task you've been avoiding",
            "Your dopamine tank is full — tackle the most challenging item on your list",
            "Ride this wave! Start the task you've been dreading — you've got the fuel for it",
        ]
        return random.choice(suggestions)
    return "Steady state. Keep going at your current pace."


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def dopamine_manager_node(state: dict) -> dict:
    """Calculate dopamine transactions and generate motivation insights."""

    economy = state.get("dopamine_economy", {})
    balance = economy.get("daily_balance", 100) if economy else 100
    transactions = economy.get("transactions", []) if economy else []

    intent = state.get("intent", "general_chat")
    current_task = state.get("current_task", {})
    task_type = current_task.get("task_type", "general") if current_task else "general"
    pattern_output = state.get("pattern_output", "")
    cognitive_output = state.get("cognitive_output", "")
    cog = state.get("cognitive_state", {})
    energy = cog.get("energy_level", 7) if cog else 7

    now = datetime.now().isoformat()
    new_transactions = []

    # ---- Detect transaction events from current interaction ----

    # Task started
    if intent == "start_task" and current_task:
        pts = TRANSACTIONS["task_started"]
        # Bonus multiplier for starting when tired
        if energy < 5:
            pts = int(pts * 1.5)
        new_transactions.append(DopamineTransaction(
            event_type="task_started", points=pts, timestamp=now,
            description=f"Started: {current_task.get('description', 'task')[:40]}"
        ).model_dump())

    # Pattern interrupted (agent detected and intervened)
    if pattern_output:
        pts = TRANSACTIONS["pattern_interrupted"]
        new_transactions.append(DopamineTransaction(
            event_type="pattern_interrupted", points=pts, timestamp=now,
            description="Broke a negative loop — that takes real effort!"
        ).model_dump())

    # Small milestone (check_in intent often means progress)
    if intent == "check_in":
        pts = TRANSACTIONS["small_milestone"]
        new_transactions.append(DopamineTransaction(
            event_type="small_milestone", points=pts, timestamp=now,
            description="Checked in — staying engaged is a win"
        ).model_dump())

    # Took break before crash (cognitive predictor warned + user listened)
    if intent == "take_break" and cognitive_output:
        pts = TRANSACTIONS["took_break_before_crash"]
        new_transactions.append(DopamineTransaction(
            event_type="took_break_before_crash", points=pts, timestamp=now,
            description="Smart break — preventing a crash is self-care"
        ).model_dump())

    # ---- Apply transactions ----
    for t in new_transactions:
        balance += t["points"]
    balance = max(0, min(100, balance))  # clamp 0-100

    all_transactions = transactions + new_transactions

    # ---- Generate reward schedule ----
    est_dur = current_task.get("estimated_duration", 60) if current_task else 60
    schedule = _generate_variable_schedule(est_dur)

    # ---- Forecast ----
    forecast = _compute_forecast(balance)
    recommendation = _get_recommendation(balance, task_type)

    # ---- Build human-readable output ----
    parts = [
        f"💰 **Dopamine Balance**: {balance}/100",
        f"📊 **Forecast**: {forecast}",
        f"💡 **Recommendation**: {recommendation}",
    ]
    if new_transactions:
        parts.append("📝 **Recent**:")
        for t in new_transactions:
            sign = "+" if t["points"] > 0 else ""
            parts.append(f"   {sign}{t['points']} — {t['description']}")
    if schedule:
        parts.append(f"🎁 **Next rewards at**: {', '.join(str(m) + ' min' for m in schedule[:4])}")

    dopamine_output = "\n".join(parts)

    updated_economy = {
        "daily_balance": balance,
        "transactions": all_transactions[-20:],  # keep last 20
        "forecast": forecast,
        "next_reward_minutes": schedule[0] if schedule else 0,
        "reward_schedule": schedule,
    }

    return {
        "dopamine_economy": updated_economy,
        "dopamine_output": dopamine_output,
    }
