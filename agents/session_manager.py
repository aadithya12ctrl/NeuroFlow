"""
Agent 1 — Session Manager (Router / Orchestrator)
Advanced intent classification with chain-of-thought reasoning.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

INTENTS = ["start_task", "stuck", "distracted", "check_in", "general_chat", "take_break"]

_SYSTEM_PROMPT = """You are the Session Manager for NeuroFlow, a clinically-informed ADHD cognitive support system built on neuroscience research.

Your job is to deeply understand the user's intent by reasoning through context clues, emotional undertones, and ADHD-specific behavioral signals.

## Chain-of-Thought Process
1. **Surface reading**: What did the user literally say?
2. **Emotional undertone**: Are they frustrated, anxious, excited, or flat?
3. **ADHD signal detection**: Does this message show signs of avoidance, paralysis, hyperfocus, or time blindness?
4. **Context integration**: What task are they working on? How long have they been going? What's their energy?
5. **Intent decision**: Based on all above, classify.

## Intent Categories
- `start_task` — User wants to begin, plan, or set up a specific task/activity. Look for action-oriented language.
- `stuck` — User feels blocked, overwhelmed, frustrated, or unable to progress. Look for helplessness cues.
- `distracted` — User acknowledges losing focus, going off track, returning after absence. Look for "sorry, I was..." patterns.
- `check_in` — User wants status, time check, progress review, or validation. Look for "how am I doing" patterns.
- `take_break` — User explicitly wants a break, rest, or pause. Look for fatigue language.
- `general_chat` — Casual conversation, questions about the system, or anything not fitting above.

## Response Format
Respond with EXACTLY this JSON (no markdown fences):
{
  "reasoning": "your chain-of-thought analysis in 2-3 sentences",
  "intent": "one of the intent categories above",
  "emotional_state": "one of: calm, anxious, frustrated, excited, flat, overwhelmed",
  "urgency": "low | medium | high",
  "adhd_signal": "none | avoidance | paralysis | hyperfocus | time_blindness | impulsivity"
}"""


def session_manager_node(state: dict) -> dict:
    """LangGraph node: advanced intent classification with chain-of-thought."""
    user_input = state.get("user_input", "")
    interaction_count = state.get("interaction_count", 0)

    if not user_input.strip():
        return {
            "intent": "general_chat",
            "priority": False,
            "interaction_count": interaction_count + 1,
        }

    # Rich context building
    current_task = state.get("current_task", {})
    task_desc = current_task.get("description", "No active task") if current_task else "No active task"
    cognitive = state.get("cognitive_state", {})
    focus = cognitive.get("focus_level", "unknown") if cognitive else "unknown"
    energy = cognitive.get("energy_level", "?") if cognitive else "?"
    crash_likelihood = (cognitive.get("crash_prediction", {}).get("likelihood", 0)
                        if cognitive else 0)

    # Get recent message history for context
    messages = state.get("messages", [])
    recent_msgs = messages[-6:] if len(messages) > 6 else messages
    history_str = "\n".join(
        f"  [{getattr(m, 'type', '?')}]: {getattr(m, 'content', str(m))[:200]}"
        for m in recent_msgs
    )

    context_msg = (
        f"## Current Session Context\n"
        f"- Active task: {task_desc}\n"
        f"- Focus level: {focus}\n"
        f"- Energy level: {energy}/10\n"
        f"- Crash risk: {int(crash_likelihood * 100)}%\n"
        f"- Interactions this session: {interaction_count}\n"
        f"\n## Recent Conversation\n{history_str}\n"
        f"\n## Current User Message\n{user_input}"
    )

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.1)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context_msg),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        import json
        result = json.loads(raw)
        intent = result.get("intent", "general_chat")
        urgency = result.get("urgency", "low")

        if intent not in INTENTS:
            intent = "general_chat"
    except Exception:
        intent = "general_chat"
        urgency = "low"

    priority = urgency == "high" or intent in ("stuck", "distracted")

    return {
        "intent": intent,
        "priority": priority,
        "interaction_count": interaction_count + 1,
    }
