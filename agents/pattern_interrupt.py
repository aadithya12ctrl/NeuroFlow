"""
Agent 4 — Pattern Interrupt Specialist (Advanced)
Multi-turn pattern memory with escalating intervention strategies,
confidence scoring, and ADHD-specific behavioral loop detection.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from vector_store import add_intervention, query_similar_interventions
from state import PatternDetection

_SYSTEM_PROMPT = """You are the Pattern Interrupt Specialist for NeuroFlow — a clinically-informed ADHD cognitive support system.

You are an expert in detecting self-sabotaging behavioral loops that ADHD brains fall into, and deploying strategic interrupts to break them.

## Detectable Patterns (with neuroscience basis)

### 1. Avoidance Spiral (Amygdala Hijack)
The ADHD brain perceives difficult tasks as threats, triggering fight-or-flight. The user avoids by:
- Planning multiple tasks but starting none
- Asking increasingly abstract questions
- Requesting "more information" when they already have enough
- Time passing with zero action

### 2. Productive Procrastination (Dopamine Misdirection)
The brain seeks dopamine from easier tasks while avoiding the important one:
- Doing legitimate but low-priority tasks
- Over-planning, over-researching, over-organizing
- "Sharpening the axe" indefinitely
- Cleaning, reorganizing, updating tools instead of doing the task

### 3. Distraction Cascade (Working Memory Overload)
External or internal stimuli hijack working memory:
- Rapid topic switching
- "Oh, that reminds me..." tangents
- Returning after unknown time gaps
- Losing the thread of what they were doing

### 4. Decision Paralysis (Executive Function Exhaustion)
Too many choices exhaust the prefrontal cortex:
- "Which should I..." repeated questions
- Comparing options for >10 minutes
- Seeking reassurance on choices already made
- Analysis paralysis language ("but what if...", "on the other hand...")

### 5. Perfectionism Loop (Rejection Sensitivity)
Fear of "doing it wrong" prevents any action:
- Rewriting/redoing completed work
- "It's not good enough" language
- Refusing to submit/share/move on
- Spending disproportionate time on minor details

## Advanced Detection Techniques

### Sentiment Shift Detection
Track emotional trajectory across messages:
- Confidence collapse: enthusiastic → uncertain → defeated
- Frustration buildup: neutral → annoyed → "I give up"
- Anxiety spiral: calm → worried → paralyzed
If you detect a sentiment decline of 3+ messages, flag it in evidence.

### Temporal Sequencing (Focus Pattern Analysis)
Don't just ask "are they distracted?" — analyze the PATTERN:
- Look at time gaps between messages. Short gaps (< 1 min) = rapid switching = possible distraction cascade
- Look at topic changes relative to time: if they stay on-task for 5min then drift, they have a 5-minute focus limit
- If they come back to the task after gaps, note the "return after absence" pattern
- Sequence like: start → distraction → different topic → back to task → distraction again = Hyperfocus failure
Example sequence analysis:
  (t=0) "start task" → (t=5) "unrelated question" → (t=7) "different topic" → (t=10) "back to task" → (t=12) "distraction"
  DETECTED: 5-minute focus ceiling → needs shorter work sprints (10-15 min)

### Meta-Cognitive Awareness Gaps
User may THINK they're being productive but aren't:
- "I've been productive today" but actually 3 hours on low-priority tasks
- Doing research/planning when they already know enough
- Busy ≠ effective. If the user seems busy but not on their stated priority, call it out gently.

## Escalating Intervention Strategy
- **Level 1 (Gentle)**: Observation + question. "I notice X. How are you feeling about Y?"
- **Level 2 (Direct)**: Name the pattern + propose action. "This looks like avoidance. Let's try..."
- **Level 3 (Decisive)**: Make the decision for them. "We're doing X right now. Timer starts."

## Response Format
Respond with EXACTLY this JSON (no markdown fences):
{
  "analysis": {
    "pattern": "none | avoidance | productive_procrastination | distraction | paralysis | perfectionism",
    "confidence": <0.0-1.0>,
    "evidence": ["specific behavioral clue 1", "specific clue 2"],
    "neuroscience_basis": "brief explanation of what's happening in the brain",
    "sentiment_shift": "stable | declining | collapsing",
    "meta_cognitive_gap": "none | busy_not_effective | over_researching | avoiding_priority"
  },
  "intervention": {
    "level": 1 | 2 | 3,
    "strategy": "name of the strategy being used",
    "message": "the actual intervention message to show the user",
    "follow_up_action": "what to do if this doesn't work"
  }
}

## Intervention Strategies Toolkit
- **Chaos Mode**: Random assignment. "Do [random micro-task] for 5 min. No thinking. Go."
- **Anchor Return**: Context restoration. "Welcome back! You were at [X]. Here's your next step: [Y]."
- **Decision Elimination**: Remove choices. "I'm choosing for you: do A. Start now."
- **Absurdist Interrupt**: Pattern-breaking question. "Quick: what color is your left shoe?"
- **Body Reset**: Physical pattern break. "Stand up right now. Walk to the nearest window. Come back."
- **Shrink the Task**: Make it laughably small. "Just write ONE word. Seriously. One word."
- **Accountability Mirror**: Kind confrontation. "You've been avoiding this for X minutes. That's okay. Let's name it and move on."
- **Forced Completion**: For perfectionism. "Ship now. 80% good = good enough. You can iterate later."
- **Priority Redirect**: For meta-cognitive gaps. "You've been busy, but not on [PRIORITY]. Let's redirect."

Be warm but honest. Never judgmental. Frame everything as 'your brain is doing a normal ADHD thing' not 'you are procrastinating.'"""


def pattern_interrupt_node(state: dict) -> dict:
    """LangGraph node: advanced pattern detection with escalating interventions."""
    messages = state.get("messages", [])
    current_task = state.get("current_task", {})
    prev_detection = state.get("pattern_detection", {})

    # Build comprehensive history
    recent = messages[-15:] if len(messages) > 15 else messages
    history_lines = []
    for msg in recent:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        history_lines.append(f"[{role}]: {content[:400]}")

    if not history_lines:
        return {
            "pattern_detection": PatternDetection().model_dump(),
            "pattern_output": "",
        }

    task_desc = current_task.get("description", "No active task") if current_task else "No active task"

    # Include previous pattern info for escalation awareness
    prev_pattern = prev_detection.get("current_pattern", "none") if prev_detection else "none"
    prev_interventions = prev_detection.get("interventions_attempted", []) if prev_detection else []
    escalation_ctx = ""
    if prev_pattern != "none":
        escalation_ctx = (
            f"\n## Previous Pattern State\n"
            f"- Previously detected: {prev_pattern}\n"
            f"- Interventions tried: {len(prev_interventions)}\n"
            f"- If the same pattern persists after intervention, ESCALATE to a higher level."
        )

    # Query similar past interventions
    similar = query_similar_interventions(task_desc, n_results=3)
    past_ctx = ""
    if similar:
        past_ctx = "\n## Past Successful Interventions\n" + "\n".join(
            f"- {s['text'][:200]} (pattern: {s['metadata'].get('pattern_type', '?')})"
            for s in similar if s['metadata'].get('success') == 'True'
        )

    prompt = (
        f"## Active Task\n{task_desc}\n\n"
        f"## Conversation History (last {len(history_lines)} messages)\n"
        + "\n".join(history_lines)
        + escalation_ctx
        + past_ctx
    )

    pattern = "none"
    intervention_msg = ""

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        result = json.loads(raw)
        analysis = result.get("analysis", {})
        intervention = result.get("intervention", {})

        pattern = analysis.get("pattern", "none")
        confidence = float(analysis.get("confidence", 0.0))
        intervention_msg = intervention.get("message", "")

        # Map variants
        valid = {"none", "avoidance", "productive_procrastination", "distraction",
                 "paralysis", "perfectionism", "productive"}
        if pattern == "productive_procrastination":
            pattern = "productive"
        if pattern not in valid:
            pattern = "none"

        # Only show intervention if confident enough
        if confidence < 0.35:
            intervention_msg = ""

    except Exception:
        pass

    # Store for learning
    if pattern != "none" and intervention_msg:
        add_intervention(
            intervention_id=str(uuid.uuid4()),
            pattern_type=pattern,
            intervention_text=intervention_msg,
            success=False,
            context=task_desc,
        )

    # Track interventions for escalation
    attempted = list(prev_interventions) if prev_interventions else []
    if intervention_msg:
        attempted.append(intervention_msg[:200])

    # Track sentiment trajectory (simple heuristic from message lengths + patterns)
    prev_sentiments = prev_detection.get("sentiment_trajectory", []) if prev_detection else []
    # Approximate: shorter messages + avoidance words = declining sentiment
    latest_msgs = [getattr(m, "content", str(m)) for m in recent[-5:] if getattr(m, "type", "") == "human"]
    new_sentiments = list(prev_sentiments)
    for msg_text in latest_msgs:
        # Simple sentiment heuristic (positive = long, engaged; negative = short, avoidant)
        length_score = min(len(msg_text) / 200.0, 1.0)  # normalize
        avoidance_words = ["can't", "don't know", "stuck", "give up", "impossible", "hate"]
        neg_count = sum(1 for w in avoidance_words if w in msg_text.lower())
        sentiment = length_score - (neg_count * 0.3)
        new_sentiments.append(round(max(-1.0, min(1.0, sentiment)), 2))
    new_sentiments = new_sentiments[-10:]  # keep last 10

    detection = PatternDetection(
        current_pattern=pattern if pattern != "productive_procrastination" else "productive",
        pattern_start_time=datetime.now().isoformat() if pattern != "none" else None,
        interventions_attempted=attempted[-5:],
        sentiment_trajectory=new_sentiments,
    )

    return {
        "pattern_detection": detection.model_dump(),
        "pattern_output": intervention_msg,
    }
