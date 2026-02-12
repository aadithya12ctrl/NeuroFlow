"""
NeuroFlow — LangGraph StateGraph Orchestration (v3.1)
Advanced 7-agent architecture demonstrating:
  1. Cyclic Workflows       — Pattern escalation loop (agent re-evaluates & retries)
  2. Conditional Edges       — Multiple routing points based on agent outputs
  3. Human-in-the-Loop       — Task plan approval gate before execution
  4. Parallel Fan-out/Fan-in — Concurrent agent execution
  5. Self-Correction Loop    — Quality gate with response retry
"""

from __future__ import annotations

import json
import traceback

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from state import NeuroFlowState
from agents.session_manager import session_manager_node
from agents.cognitive_predictor import cognitive_predictor_node
from agents.context_architect import context_architect_node
from agents.pattern_interrupt import pattern_interrupt_node
from agents.time_reality import time_reality_node
from agents.focus_builder import focus_builder_node
from agents.dopamine_manager import dopamine_manager_node


# ============================================================
# Response Generator Node
# ============================================================

_RESPONSE_SYSTEM = """You are NeuroFlow 🧠 — a warm, knowledgeable ADHD cognitive coach.

Your personality:
- You speak like a supportive friend who deeply understands ADHD neuroscience
- You normalize ADHD struggles ("your brain does this because...") without excusing inaction
- You're direct when needed but never harsh
- You use emojis meaningfully, not excessively
- You celebrate wins genuinely — even tiny ones
- You reference neuroscience casually ("dopamine loves novelty, so let's use that...")

## Synthesis Rules
1. If a Context Package was generated, present it prominently — it's the main content
2. If a cognitive alert exists, weave it in naturally at the beginning or end
3. If a pattern intervention exists, address it with empathy before anything else
4. If time awareness info exists, include it contextually
5. If a focus environment was configured, mention the setup (music, body double, timer)
6. If dopamine economy info exists, show the balance and any recommendations
7. If nothing special was generated, respond naturally and helpfully to the user's message
8. NEVER mention "agents", "nodes", "graph", or system internals
9. Keep responses focused — don't ramble. ADHD users lose interest quickly.
10. Use markdown formatting for structure (headers, bold, lists)
11. End with a clear next action when appropriate"""


def response_generator_node(state: dict) -> dict:
    """Synthesise all agent outputs into one cohesive, personality-rich response."""
    user_input = state.get("user_input", "")
    intent = state.get("intent", "general_chat")
    cognitive_output = state.get("cognitive_output", "")
    context_output = state.get("context_output", "")
    pattern_output = state.get("pattern_output", "")
    time_output = state.get("time_output", "")
    focus_output = state.get("focus_output", "")
    dopamine_output = state.get("dopamine_output", "")

    # Build synthesis prompt
    parts = []
    if pattern_output:
        parts.append(f"## Pattern Intervention (HIGH PRIORITY — address this first)\n{pattern_output}")
    if context_output:
        parts.append(f"## Context Package (present this prominently)\n{context_output}")
    if focus_output:
        parts.append(f"## Focus Environment Setup\n{focus_output}")
    if cognitive_output:
        parts.append(f"## Cognitive Alert\n{cognitive_output}")
    if time_output:
        parts.append(f"## Time Awareness\n{time_output}")
    if dopamine_output:
        parts.append(f"## Dopamine Economy\n{dopamine_output}")

    combined = "\n\n---\n\n".join(parts) if parts else ""

    # Cognitive state for context
    cog = state.get("cognitive_state", {})
    focus = cog.get("focus_level", "?") if cog else "?"
    energy = cog.get("energy_level", "?") if cog else "?"

    # Dopamine balance
    economy = state.get("dopamine_economy", {})
    dopamine_bal = economy.get("daily_balance", "?") if economy else "?"

    prompt = (
        f"## User's Message\n{user_input}\n\n"
        f"## Detected Intent: {intent}\n"
        f"## User's State: Focus={focus}, Energy={energy}/10, Dopamine={dopamine_bal}/100\n\n"
        f"## Agent Outputs to Synthesize\n{combined}\n\n"
        "Create a single, natural response. If there's a context package, include all "
        "its details formatted clearly. If there's a focus environment, mention what's "
        "being set up (music, body double, timer). If there's dopamine info, weave in "
        "the balance and recommendation naturally. Be warm, direct, and ADHD-friendly."
    )

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=_RESPONSE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        final = response.content.strip()
    except Exception as e:
        print(f"[NeuroFlow] Response generator error: {e}")
        if context_output:
            final = f"Here's your focus plan! 🎯\n\n{context_output}"
        elif pattern_output:
            final = pattern_output
        elif focus_output:
            final = f"🎵 Environment ready!\n\n{focus_output}"
        elif cognitive_output:
            final = cognitive_output
        elif dopamine_output:
            final = dopamine_output
        elif time_output:
            final = time_output
        else:
            final = (
                "Hey! 👋 I'm here to help you navigate your tasks. "
                "Tell me what you'd like to work on."
            )

    return {
        "response": final,
        "messages": [AIMessage(content=final)],
        "cognitive_output": "",
        "context_output": "",
        "pattern_output": "",
        "time_output": "",
        "focus_output": "",
        "dopamine_output": "",
    }


# ============================================================
# FEATURE 5: Self-Correction Loop — Quality Gate
# ============================================================

def quality_gate_node(state: dict) -> dict:
    """Evaluate the generated response quality.
    
    Creates a SELF-CORRECTION CYCLE:
    response_generator → quality_gate → [retry → response_generator] or [→ END]
    """
    response = state.get("response", "")
    retry_count = state.get("response_retry_count", 0)
    
    quality_issues = []
    if not response or len(response.strip()) < 30:
        quality_issues.append("Response too short")
    if response and any(phrase in response.lower() for phrase in [
        "i'm having trouble", "something went wrong", "please try again", "error"
    ]):
        quality_issues.append("Response contains error language")
    
    score = 1.0 if not quality_issues else (0.4 if len(quality_issues) == 1 else 0.2)
    
    return {
        "quality_score": score,
        "response_retry_count": retry_count,
    }


# ============================================================
# FEATURE 3: Human-in-the-Loop — Plan Approval Gate
# ============================================================

def human_approval_gate_node(state: dict) -> dict:
    """Human-in-the-loop approval point for task plans.
    
    The graph compiles with interrupt_before=["human_approval_gate"],
    which pauses execution here. The calling code (run_agent) detects
    the interrupt and resumes after approval.
    
    For complex tasks (>5 steps or >60min), this flags for review.
    """
    current_task = state.get("current_task", {})
    needs_approval = False
    if current_task:
        steps = current_task.get("context_package", {}).get("micro_steps", [])
        estimated = current_task.get("realistic_duration", 0)
        if len(steps) > 5 or estimated > 60:
            needs_approval = True
    
    return {"needs_human_approval": needs_approval}


# ============================================================
# FEATURE 1: Cyclic Workflow — Pattern Escalation Node
# ============================================================

def pattern_escalation_node(state: dict) -> dict:
    """Increment escalation level and loop back to pattern_interrupt.
    
    CYCLIC WORKFLOW: pattern_interrupt → severity_router → escalation → pattern_interrupt
    
    Each loop increases intervention strategy:
      Level 0: Gentle observation + question
      Level 1: Direct naming + proposed action  
      Level 2: Decisive — make the decision for the user (max, exits loop)
    """
    current_level = state.get("pattern_escalation_level", 0)
    pattern = state.get("pattern_detection", {})
    
    user_input = state.get("user_input", "")
    escalation_context = (
        f"\n\n[SYSTEM: ESCALATING to Level {current_level + 1}. "
        f"Previous: {pattern.get('current_pattern', 'unknown')}. "
        f"Use MORE DIRECT intervention strategy.]"
    )
    
    return {
        "pattern_escalation_level": current_level + 1,
        "user_input": user_input + escalation_context,
    }


# ============================================================
# FEATURE 5: Self-Correction — Retry Preparation Node
# ============================================================

def response_retry_node(state: dict) -> dict:
    """Prepare for response re-generation (self-correction loop).
    Increments retry counter so the quality gate won't loop forever.
    """
    return {
        "response_retry_count": state.get("response_retry_count", 0) + 1,
    }


# ============================================================
# ROUTING FUNCTIONS (3 Conditional Edge Points)
# ============================================================

# CONDITIONAL EDGE #1: Intent-based routing from Session Manager
def _route_intent(state: dict) -> str:
    """Route based on detected user intent."""
    intent = state.get("intent", "general_chat")
    if intent == "start_task":
        return "start_task"
    elif intent in ("stuck", "distracted"):
        return "stuck_or_distracted"
    elif intent in ("check_in", "take_break"):
        return "check_in"
    return "general_chat"


# CONDITIONAL EDGE #2: Pattern severity routing (enables CYCLES)
def _route_pattern_severity(state: dict) -> str:
    """Route based on pattern detection confidence + escalation level.
    
    HIGH confidence + not maxed → CYCLE back to pattern_interrupt
    MEDIUM confidence → full cognitive analysis pipeline
    LOW confidence → skip to response generator
    """
    pattern = state.get("pattern_detection", {})
    escalation = state.get("pattern_escalation_level", 0)
    
    confidence = 0.0
    if isinstance(pattern, dict):
        confidence = pattern.get("confidence", 0.0)
        if confidence == 0.0 and pattern.get("current_pattern", "none") != "none":
            confidence = 0.6
    
    if confidence > 0.7 and escalation < 2:
        return "escalate"       # → CYCLE back to pattern_interrupt
    if confidence > 0.3:
        return "full_analysis"  # → cognitive_predictor → full pipeline
    return "quick_response"     # → response_generator (low severity)


# CONDITIONAL EDGE #3: Quality gate routing (enables SELF-CORRECTION)
def _route_quality(state: dict) -> str:
    """Route based on response quality. Retry once if quality is poor."""
    score = state.get("quality_score", 1.0)
    retry_count = state.get("response_retry_count", 0)
    
    if score < 0.5 and retry_count < 1:
        return "retry"   # → CYCLE back to response_generator
    return "accept"      # → END


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph():
    """Construct the NeuroFlow StateGraph with advanced agentic patterns.
    
    Architecture:
    
    ┌─────────────────┐
    │ session_manager  │ (entry point)
    └────────┬────────┘
             │ CONDITIONAL EDGE #1 (intent routing)
             ├── start_task ──────────► context_architect
             │                              │
             │                    human_approval_gate ◄── HUMAN-IN-THE-LOOP
             │                         │         │          (interrupt_before)
             │                    focus_builder  cognitive_predictor  ◄── PARALLEL FAN-OUT
             │                         │         │
             │                    dopamine_manager ◄── FAN-IN (both edges converge)
             │
             ├── stuck/distracted ───► pattern_interrupt
             │                              │
             │                    CONDITIONAL EDGE #2 (severity)
             │                    ├── escalate ──► pattern_escalation ──┐
             │                    │                                      │
             │                    │    ┌────────────────────────────────┘
             │                    │    └──► pattern_interrupt  ◄── CYCLIC WORKFLOW
             │                    │
             │                    ├── full_analysis ──► cognitive_predictor ──► dopamine_manager
             │                    └── quick_response ──► response_generator
             │
             ├── check_in ───────► time_reality ──► cognitive_predictor ──► dopamine_manager
             │
             └── general_chat ──► cognitive_predictor ──► dopamine_manager
    
    All paths converge at:
    dopamine_manager ──► response_generator ──► quality_gate
                                                    │
                                          CONDITIONAL EDGE #3 (quality)
                                          ├── retry ──► response_retry ──► response_generator ◄── SELF-CORRECTION
                                          └── accept ──► END
    """
    graph = StateGraph(NeuroFlowState)

    # ── Register all nodes ──
    graph.add_node("session_manager", session_manager_node)
    graph.add_node("context_architect", context_architect_node)
    graph.add_node("human_approval_gate", human_approval_gate_node)
    graph.add_node("focus_builder", focus_builder_node)
    graph.add_node("cognitive_predictor", cognitive_predictor_node)
    graph.add_node("pattern_interrupt", pattern_interrupt_node)
    graph.add_node("pattern_escalation", pattern_escalation_node)
    graph.add_node("time_reality", time_reality_node)
    graph.add_node("dopamine_manager", dopamine_manager_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("response_retry", response_retry_node)

    # ── Entry Point ──
    graph.set_entry_point("session_manager")

    # ══════════════════════════════════════════════════════════
    # CONDITIONAL EDGE #1: Intent-based routing (4 paths)
    # ══════════════════════════════════════════════════════════
    graph.add_conditional_edges(
        "session_manager",
        _route_intent,
        {
            "start_task": "context_architect",
            "stuck_or_distracted": "pattern_interrupt",
            "check_in": "time_reality",
            "general_chat": "cognitive_predictor",
        },
    )

    # ══════════════════════════════════════════════════════════
    # START_TASK PATH (with Human-in-Loop + Parallel Fan-out)
    # ══════════════════════════════════════════════════════════
    
    # context_architect → human_approval_gate (HITL interrupt point)
    graph.add_edge("context_architect", "human_approval_gate")
    
    # PARALLEL FAN-OUT: approval gate forks to both agents concurrently
    # LangGraph runs both focus_builder and cognitive_predictor in parallel
    # because they share the same source node
    graph.add_edge("human_approval_gate", "focus_builder")
    graph.add_edge("human_approval_gate", "cognitive_predictor")
    
    # FAN-IN: Both parallel branches converge at dopamine_manager
    # dopamine_manager waits for both to complete before executing
    graph.add_edge("focus_builder", "dopamine_manager")
    graph.add_edge("cognitive_predictor", "dopamine_manager")

    # ══════════════════════════════════════════════════════════
    # STUCK/DISTRACTED PATH (with Cyclic Escalation)  
    # ══════════════════════════════════════════════════════════
    
    # CONDITIONAL EDGE #2: Severity-based routing after pattern detection
    graph.add_conditional_edges(
        "pattern_interrupt",
        _route_pattern_severity,
        {
            "escalate": "pattern_escalation",       # → CYCLE
            "full_analysis": "dopamine_manager",     # → full pipeline (skip cognitive since pattern already analysed)
            "quick_response": "response_generator",  # → direct response
        },
    )
    
    # CYCLIC WORKFLOW: escalation loops back to pattern_interrupt
    graph.add_edge("pattern_escalation", "pattern_interrupt")

    # ══════════════════════════════════════════════════════════
    # CHECK-IN PATH
    # ══════════════════════════════════════════════════════════
    graph.add_edge("time_reality", "dopamine_manager")

    # ══════════════════════════════════════════════════════════
    # CONVERGENCE → Response → Quality Gate
    # ══════════════════════════════════════════════════════════
    graph.add_edge("dopamine_manager", "response_generator")
    graph.add_edge("response_generator", "quality_gate")
    
    # CONDITIONAL EDGE #3: Quality gate (self-correction loop)
    graph.add_conditional_edges(
        "quality_gate",
        _route_quality,
        {
            "retry": "response_retry",
            "accept": END,
        },
    )
    
    # SELF-CORRECTION CYCLE: retry → response_generator (loop back)
    graph.add_edge("response_retry", "response_generator")

    # ── Compile with memory + Human-in-the-Loop ──
    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_approval_gate"],
    )
