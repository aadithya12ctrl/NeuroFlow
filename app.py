"""
NeuroFlow — Streamlit Application (v3.0)
ADHD Cognitive Support System with 7-Agent Architecture,
Focus Environment Builder, and Dopamine Economy Manager.
"""

import os
import uuid
import traceback
import time
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(
    page_title="NeuroFlow — ADHD Cognitive Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from graph import build_graph
from state import (
    CognitiveState,
    InteractionMetrics,
    PatternDetection,
    TaskInfo,
    UserPreferences,
    TaskEnvironment,
)
from database import init_db, log_interaction

init_db()

# ============================================================
# CSS — Warm Autumn Design System
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --cream: #F5F0E8;
    --cream-light: #FAF7F2;
    --taupe: #C9B8A8;
    --taupe-light: #DDD3C7;
    --olive: #8FA67E;
    --olive-dark: #6E8A5E;
    --olive-light: #A8BF9A;
    --burnt-orange: #C8763F;
    --burnt-orange-light: #D89A6A;
    --rust: #A84C32;
    --terracotta: #B85C4F;
    --dark-brown: #3E3028;
    --medium-brown: #6B5B52;
    --warm-shadow: rgba(62, 48, 40, 0.08);
    --warm-shadow-md: rgba(62, 48, 40, 0.12);
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--cream) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #000000 !important;
}
.stApp * {
    color: #000000;
}
.stApp > header { background: transparent !important; }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3E3028 0%, #4A3C32 50%, #3E3028 100%) !important;
    border-right: 1px solid rgba(200,118,63,0.2);
}
[data-testid="stSidebar"] * {
    color: var(--cream) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: var(--taupe-light) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    margin-bottom: 0.3rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(200,118,63,0.2) !important;
    border-color: rgba(200,118,63,0.4) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(200,118,63,0.25) !important;
    border-color: var(--burnt-orange) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--taupe); border-radius: 3px; }

/* Card */
.nf-card {
    background: var(--cream-light);
    border: 1px solid var(--taupe-light);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px var(--warm-shadow);
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease;
}
.nf-card:hover { box-shadow: 0 4px 16px var(--warm-shadow-md); }

/* Metric cards */
.metric-card {
    background: var(--cream-light);
    border: 1px solid var(--taupe-light);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px var(--warm-shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 14px var(--warm-shadow-md); }
.metric-icon { font-size: 1.6rem; margin-bottom: 0.2rem; }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 600; color: #000000;
}
.metric-label {
    font-size: 0.75rem; color: #333333;
    text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.15rem;
}

/* Focus pills */
.focus-pill {
    display: inline-block; padding: 0.25rem 0.9rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-top: 0.3rem;
}
.focus-low { background: rgba(168,76,50,0.15); color: var(--rust); }
.focus-medium { background: rgba(200,118,63,0.15); color: var(--burnt-orange); }
.focus-high { background: rgba(143,166,126,0.15); color: var(--olive-dark); }
.focus-hyperfocus { background: rgba(143,100,180,0.15); color: #7B5EA7; }

/* Progress bar */
.progress-bar-container {
    background: var(--taupe-light); border-radius: 8px; height: 22px;
    overflow: hidden; margin: 0.5rem 0; position: relative;
}
.progress-bar-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, var(--olive), var(--olive-light));
    transition: width 0.6s ease; display: flex; align-items: center;
    justify-content: flex-end; padding-right: 8px;
}
.progress-bar-fill span { font-size: 0.7rem; font-weight: 600; color: white; }

/* Section headers */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #000000;
    margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;
}
.page-title {
    font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem;
    background: linear-gradient(135deg, var(--olive-dark), var(--burnt-orange));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-subtitle {
    color: var(--medium-brown); font-size: 0.9rem; margin-bottom: 1.2rem;
}

/* Bars */
.bar-track { background: var(--taupe-light); border-radius: 6px; height: 10px; overflow: hidden; margin-top: 0.3rem; }
.bar-fill-energy { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--burnt-orange), var(--burnt-orange-light)); transition: width 0.5s ease; }
.bar-fill-dopamine { height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--rust), var(--terracotta)); transition: width 0.5s ease; }
.bar-fill-crash {
    height: 100%; border-radius: 6px; transition: width 0.5s ease;
}

/* Next step pill */
.next-step-pill {
    background: rgba(143,166,126,0.18); color: var(--olive-dark);
    padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.85rem;
    font-weight: 500; display: inline-block; margin-top: 0.3rem;
}

/* Agent info box */
.agent-info {
    background: rgba(143,166,126,0.08);
    border: 1px solid rgba(143,166,126,0.2);
    border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
    font-size: 0.85rem; line-height: 1.6;
}
.agent-info strong { color: var(--olive-dark); }

/* Streamlit overrides */
[data-testid="stChatMessage"] {
    background: var(--cream-light) !important;
    border: 1px solid var(--taupe-light) !important;
    border-radius: 14px !important;
    margin-bottom: 0.5rem !important;
    box-shadow: 0 1px 4px var(--warm-shadow) !important;
}
.stButton > button {
    background: var(--olive-dark) !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    padding: 0.55rem 1.3rem !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--olive) !important; transform: translateY(-1px);
    box-shadow: 0 4px 10px var(--warm-shadow-md) !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stChatInput"] textarea {
    background: var(--cream-light) !important;
    border: 1.5px solid var(--taupe) !important;
    border-radius: 10px !important;
    color: var(--dark-brown) !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--olive) !important;
    box-shadow: 0 0 0 2px rgba(143,166,126,0.2) !important;
}

/* Keyframes */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.pulse { animation: pulse 1.5s infinite; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================

def _init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_start = datetime.now().isoformat()
        st.session_state.chat_history = []
        st.session_state.cognitive = CognitiveState().model_dump()
        st.session_state.interaction_metrics = InteractionMetrics().model_dump()
        st.session_state.pattern_detection = PatternDetection().model_dump()
        st.session_state.current_task = {}
        st.session_state.dopamine_economy = {}
        st.session_state.graph = build_graph()
        st.session_state.interaction_count = 0
        st.session_state.last_msg_time = datetime.now()
        st.session_state.pattern_history = []
        st.session_state.parked_thoughts = []
        st.session_state.notepad_content = ""
        st.session_state.timer_active = False
        st.session_state.timer_end_time = None

_init_session()

# ============================================================
# Helper: invoke graph
# ============================================================

def run_agent(user_input: str) -> str:
    now = datetime.now()
    elapsed = (now - st.session_state.last_msg_time).total_seconds()
    metrics = InteractionMetrics(**st.session_state.interaction_metrics)
    metrics.message_lengths.append(len(user_input))
    metrics.response_times.append(elapsed)
    metrics.avg_message_length = (
        sum(metrics.message_lengths) // len(metrics.message_lengths)
        if metrics.message_lengths else 0
    )
    from utils.metrics import detect_trend
    metrics.response_time_trend = detect_trend(metrics.response_times)
    metrics.current_typing_speed = len(user_input) / max(elapsed, 1)
    if metrics.typing_speed_baseline == 0:
        metrics.typing_speed_baseline = metrics.current_typing_speed
    st.session_state.interaction_metrics = metrics.model_dump()
    st.session_state.last_msg_time = now

    try:
        log_interaction(
            typing_speed=metrics.current_typing_speed,
            message_length=len(user_input),
            response_time=elapsed,
            session_duration=(now - datetime.fromisoformat(st.session_state.session_start)).total_seconds(),
            task_id=st.session_state.current_task.get("task_id") if st.session_state.current_task else None,
        )
    except Exception:
        pass

    input_state = {
        "user_input": user_input,
        "messages": [HumanMessage(content=user_input)],
        "session_id": st.session_state.session_id,
        "session_start": st.session_state.session_start,
        "interaction_count": st.session_state.interaction_count,
        "cognitive_state": st.session_state.cognitive,
        "interaction_metrics": st.session_state.interaction_metrics,
        "pattern_detection": st.session_state.pattern_detection,
        "current_task": st.session_state.current_task,
        "dopamine_economy": st.session_state.dopamine_economy,
        "user_preferences": UserPreferences().model_dump(),
        # Advanced graph control fields
        "pattern_escalation_level": 0,   # Reset for each new interaction
        "response_retry_count": 0,       # Reset for each new interaction
        "quality_score": 1.0,            # Default high quality
        "needs_human_approval": False,   # Default no approval needed
    }
    config = {"configurable": {"thread_id": st.session_state.session_id}}

    try:
        result = st.session_state.graph.invoke(input_state, config)
        
        # ── HUMAN-IN-THE-LOOP: Handle interrupt_before ──
        # If the graph paused at human_approval_gate (interrupt_before),
        # result won't contain a response. We detect this and resume.
        if not result.get("response"):
            # Graph was interrupted — this is the human-in-the-loop pattern
            # In production, you'd show the plan and wait for approval.
            # Here we auto-approve to keep the flow seamless.
            print("[NeuroFlow] Human-in-the-loop: Auto-approving task plan")
            
            # Resume the graph from the interrupt point (pass None to continue)
            result = st.session_state.graph.invoke(None, config)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[NeuroFlow ERROR]\n{tb}")
        return f"⚠️ Something went wrong: {e}"

    if result.get("cognitive_state"):
        st.session_state.cognitive = result["cognitive_state"]
    if result.get("current_task"):
        st.session_state.current_task = result["current_task"]
    if result.get("pattern_detection"):
        st.session_state.pattern_detection = result["pattern_detection"]
        pdet = result["pattern_detection"]
        if pdet.get("current_pattern") and pdet["current_pattern"] != "none":
            st.session_state.pattern_history.append({
                "pattern": pdet["current_pattern"],
                "time": datetime.now().strftime("%I:%M %p"),
            })
    if result.get("dopamine_economy"):
        st.session_state.dopamine_economy = result["dopamine_economy"]
    st.session_state.interaction_count = result.get(
        "interaction_count", st.session_state.interaction_count
    )
    return result.get("response", "I'm having trouble responding. Please try again.")


def _time_fmt():
    return datetime.now().strftime("%I:%M %p")


# ============================================================
# Sidebar Navigation
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size: 2rem;">🧠</div>
        <div style="font-size: 1.4rem; font-weight: 800; 
             background: linear-gradient(135deg, #A8BF9A, #D89A6A, #B85C4F);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text;">NeuroFlow</div>
        <div style="font-size: 0.7rem; opacity: 0.6; margin-top: 2px;">ADHD Cognitive Support v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🎯 Focus Studio",
            "🧘 Focus Mode",
            "📊 Cognitive Monitor",
            "🔄 Pattern Detective",
            "⏱️ Time Reality",
        ],
        label_visibility="collapsed",
    )

    # Handle page override (from Pattern Detective buttons, Focus Mode exit, etc.)
    if "page_override" in st.session_state:
        page = st.session_state.pop("page_override")

    st.markdown("---")

    # Sidebar mini status
    cog = CognitiveState(**st.session_state.cognitive)
    focus_colors = {"low": "🔴", "medium": "🟡", "high": "🟢", "hyperfocus": "🟣"}

    # Dopamine economy (from agent)
    economy = st.session_state.get("dopamine_economy", {})
    dop_balance = economy.get("daily_balance", cog.dopamine_balance) if economy else cog.dopamine_balance
    dop_forecast = economy.get("forecast", "") if economy else ""

    st.markdown(f"""
    <div style="padding: 0.5rem; font-size: 0.8rem; opacity: 0.85;">
        <div style="margin-bottom: 0.4rem;">{focus_colors.get(cog.focus_level, '🟡')} Focus: <strong>{cog.focus_level.upper()}</strong></div>
        <div style="margin-bottom: 0.4rem;">⚡ Energy: <strong>{cog.energy_level}/10</strong></div>
        <div style="margin-bottom: 0.4rem;">💰 Dopamine: <strong>{dop_balance}/100</strong></div>
        <div style="margin-bottom: 0.4rem; font-size: 0.7rem;">{dop_forecast}</div>
        <div>⏱️ Session: <strong>{int((datetime.now() - datetime.fromisoformat(st.session_state.session_start)).total_seconds() / 60)}m</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Active task mini
    task = st.session_state.current_task
    if task and task.get("description"):
        st.markdown("---")
        desc = task["description"][:40] + ("..." if len(task.get("description","")) > 40 else "")
        pct = task.get("progress_percent", 0)
        st.markdown(f"""
        <div style="padding: 0.5rem; font-size: 0.78rem; opacity: 0.85;">
            <div style="font-weight: 700; margin-bottom: 0.3rem;">📋 Active Task</div>
            <div style="margin-bottom: 0.3rem;">{desc}</div>
            <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="width: {max(pct,3)}%; height: 100%; background: var(--olive-light); border-radius: 4px;"></div>
            </div>
            <div style="margin-top: 0.2rem; font-size: 0.7rem;">{pct}% complete</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: Dashboard
# ============================================================

if page == "🏠 Dashboard":
    st.markdown('<div class="page-title">🏠 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Your ADHD command center — chat with all agents at once</div>', unsafe_allow_html=True)

    # ── Metrics Row ──
    cog = CognitiveState(**st.session_state.cognitive)
    session_start = datetime.fromisoformat(st.session_state.session_start)
    elapsed_min = int((datetime.now() - session_start).total_seconds() / 60)
    time_str = f"{elapsed_min // 60}h {elapsed_min % 60}m" if elapsed_min >= 60 else f"{elapsed_min}m"
    focus_emoji = {"low": "🔴", "medium": "🟡", "high": "🟢", "hyperfocus": "🟣"}
    focus_class = f"focus-{cog.focus_level}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">{focus_emoji.get(cog.focus_level,"🟡")}</div><div class="metric-value"><span class="focus-pill {focus_class}">{cog.focus_level.upper()}</span></div><div class="metric-label">Focus State</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">⚡</div><div class="metric-value">{cog.energy_level}/10</div><div class="bar-track"><div class="bar-fill-energy" style="width:{cog.energy_level*10}%"></div></div><div class="metric-label">Energy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">🔥</div><div class="metric-value">{cog.dopamine_balance}/100</div><div class="bar-track"><div class="bar-fill-dopamine" style="width:{cog.dopamine_balance}%"></div></div><div class="metric-label">Dopamine</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">⏱️</div><div class="metric-value">{time_str}</div><div class="metric-label">Session Time</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Active Task Card ──
    task = st.session_state.current_task
    if task and task.get("description"):
        task_info = TaskInfo(**task)
        if task_info.start_time:
            task_elapsed = int((datetime.now() - datetime.fromisoformat(task_info.start_time)).total_seconds() / 60)
            remaining = max(0, task_info.estimated_duration - task_elapsed)
        else:
            task_elapsed = 0
            remaining = task_info.estimated_duration
        remaining_ms = [m for m in task_info.progress_milestones if m not in task_info.completed_milestones]
        next_ms = remaining_ms[0] if remaining_ms else "All milestones complete! 🏆"
        pct = task_info.progress_percent
        st.markdown(f"""
        <div class="nf-card">
            <div class="section-header">📋 Active Task</div>
            <div style="font-size:1.15rem;font-weight:700;">{task_info.description}</div>
            <div class="progress-bar-container"><div class="progress-bar-fill" style="width:{max(pct,3)}%"><span>{pct}%</span></div></div>
            <div class="next-step-pill">➡️ {next_ms}</div>
            <div style="margin-top:0.5rem;font-size:0.8rem;color:var(--medium-brown);">
                Started {task_elapsed}m ago • Est. {task_info.estimated_duration}m • {remaining}m left
            </div>
        </div>
        """, unsafe_allow_html=True)

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            if st.button("▶️ Go to Focus Mode", key="d_focus_mode", use_container_width=True):
                st.session_state["page_override"] = "🧘 Focus Mode"
                st.rerun()
        with tc2:
            if st.button("⏸️ Pause", key="d_pause", use_container_width=True):
                st.session_state.current_task = {}
                st.rerun()
        with tc3:
            if st.button("🆘 Help", key="d_help", use_container_width=True):
                st.session_state.pending_input = f"I'm stuck on: {task_info.description}"
                st.rerun()

    # ── Chat Interface ──
    st.markdown('<div class="section-header">💬 NeuroFlow Chat</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(
                "Hey! I'm your ADHD cognitive support companion. 🌿\n\n"
                "I can help you:\n"
                "- 🎯 **Start a task** with a custom focus plan\n"
                "- 🧘 **Enter Focus Mode** with custom music & tools\n"
                "- 🆘 **Get unstuck** when you hit a wall\n"
                "- ⏱️ **Track your time** realistically\n\n"
                "Type below or use the sidebar pages for specialized help!"
            )

    for msg in st.session_state.chat_history:
        avatar = "🧠" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Handle pending input
    if "pending_input" in st.session_state:
        pending = st.session_state.pop("pending_input")
        st.session_state.chat_history.append({"role": "user", "content": pending})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pending)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🧠 Thinking..."):
                resp = run_agent(pending)
            st.markdown(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()

    if user_input := st.chat_input("Tell me what you'd like to work on..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🧠 Thinking..."):
                resp = run_agent(user_input)
            st.markdown(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()


# ============================================================
# PAGE: Focus Studio (Context Architect)
# ============================================================

elif page == "🎯 Focus Studio":
    st.markdown('<div class="page-title">🎯 Focus Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Design your perfect task environment — the Context Architect builds a custom focus plan</div>', unsafe_allow_html=True)

    with st.form("focus_form", clear_on_submit=False):
        st.markdown("### 📝 Task Details")
        task_desc = st.text_area(
            "What do you need to work on?",
            placeholder="e.g., Write a 2000-word research paper on climate change for my biology class",
            height=100,
        )

        col1, col2 = st.columns(2)
        with col1:
            energy = st.slider("⚡ Current Energy Level", 1, 10, 5)
            time_estimate = st.number_input("⏱️ Your Time Estimate (minutes)", min_value=5, max_value=480, value=30)
        with col2:
            task_type = st.selectbox("💻 Task Type", ["General", "Coding", "Writing", "Revision"], index=0)
            difficulty = st.select_slider("🧠 How Hard Does This Feel?", options=["Easy", "Medium", "Hard", "Overwhelming"], value="Medium")
            priority_level = st.selectbox("🚦 Priority", ["Low — Whenever", "Medium — Today", "High — Urgent", "Critical — Due Soon"])

        music_genre = st.text_input(
            "🎵 Preferred Music Genre (optional)",
            placeholder="e.g., K-Pop, Lo-Fi, EDM, Classical, Jazz, Metal, Bollywood",
        )

        preferences = st.multiselect(
            "🎮 What would help you focus?",
            ["Gamification / Point System", "Music Recommendations", "Body Doubling Tips",
             "Pomodoro Timer", "Minimal Distractions Mode", "Accountability Check-ins"],
            default=["Pomodoro Timer"],
        )

        submitted = st.form_submit_button("🚀 Generate Focus Plan", use_container_width=True)

    if submitted and task_desc:
        # Build enriched prompt
        pref_str = ", ".join(preferences) if preferences else "none"
        genre_str = music_genre.strip() if music_genre else "any"
        enriched = (
            f"{task_desc}\n\n"
            f"[User Context] Energy: {energy}/10, Difficulty feeling: {difficulty}, "
            f"Task Type: {task_type.lower()}, "
            f"Priority: {priority_level}, Their time estimate: {time_estimate} minutes, "
            f"Preferred Music Genre: {genre_str}, "
            f"Preferences: {pref_str}"
        )

        with st.spinner("🧠 NeuroFlow agents are designing your focus environment..."):
            resp = run_agent(enriched)

        st.markdown("---")
        st.markdown(resp)
        st.balloons()
    
    # Show current task if exists
    task = st.session_state.current_task
    if task and task.get("description"):
        st.markdown("---")
        task_info = TaskInfo(**task)
        env = task_info.environment
        
        st.markdown(f"### 📋 Active Task: {task_info.description}")
        
        # ── All Micro-Steps as Checkboxes ──
        steps = task_info.context_package.get("micro_steps", [])
        if steps:
            st.markdown("#### ✅ Your Step-by-Step Plan")
            st.caption("Check off each step as you complete it — each checkbox = dopamine hit 🎯")
            for i, s in enumerate(steps, 1):
                step_text = s.get("step", s) if isinstance(s, dict) else str(s)
                reward = s.get("dopamine_reward", "+🧠") if isinstance(s, dict) else "+🧠"
                checked = st.checkbox(f"{step_text}  ({reward})", key=f"studio_step_{i}")
                if checked:
                    st.session_state[f"step_{i}_done"] = True
        
        # ── BPM-Mapped Playlist ──
        env_data = task.get("environment", {})
        playlist = env_data.get("playlist", [])
        if not playlist:
            # Try to extract from context package
            playlist = task_info.context_package.get("playlist", [])
        
        if playlist:
            st.markdown("#### 🎶 Your Focus Playlist (BPM-Mapped)")
            st.caption("Songs matched to each work phase — BPM follows HIGH → LOW → HIGH → LOW pattern")
            
            for track in playlist:
                if isinstance(track, dict):
                    section = track.get("section", "")
                    song = track.get("song", "")
                    bpm = track.get("bpm", "?")
                    mapped_step = track.get("mapped_step", "")
                    reason = track.get("reason", "")
                    
                    # Color-code BPM
                    try:
                        bpm_val = int(bpm)
                        if bpm_val >= 130:
                            bpm_bg = "rgba(255,107,107,0.2)"
                            bpm_border = "#ff6b6b"
                            bpm_label = "🔴 HIGH"
                        elif bpm_val >= 90:
                            bpm_bg = "rgba(255,217,61,0.2)"
                            bpm_border = "#ffd93d"
                            bpm_label = "🟡 MED"
                        else:
                            bpm_bg = "rgba(107,203,119,0.2)"
                            bpm_border = "#6bcb77"
                            bpm_label = "🟢 LOW"
                    except:
                        bpm_bg = "rgba(150,150,150,0.2)"
                        bpm_border = "#999"
                        bpm_label = "⚪"
                    
                    st.markdown(f"""<div style="background:{bpm_bg}; border-left:4px solid {bpm_border}; border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
<div style="display:flex; justify-content:space-between; align-items:center;">
<strong style="font-size:0.95rem;">{section}</strong>
<span style="background:{bpm_border}; color:#000; font-weight:700; padding:2px 10px; border-radius:12px; font-size:0.8rem;">{bpm} BPM</span>
</div>
<div style="margin-top:0.3rem; font-size:0.9rem;">🎵 {song}</div>
<div style="margin-top:0.3rem; font-size:0.78rem; opacity:0.7;">📋 {mapped_step}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"- {track}")
        
        # ── Break Activities ──
        breaks = env_data.get("break_activities", [])
        if breaks:
            st.markdown("#### 💃 Break Activities")
            for b in breaks[:4]:
                st.markdown(f"- {b}")
        
        # ── Environment Preview ──
        st.markdown(f"""
        <div class="nf-card">
            <div class="section-header">🛠️ Virtual Environment Setup</div>
            <div style="display:flex; gap:1rem; flex-wrap:wrap;">
                <span class="next-step-pill">🎵 Music: {env.music_style.replace('_', ' ').title()}</span>
                <span class="next-step-pill">⏱️ Timer: {env.timer_mode.title()}</span>
                <span class="next-step-pill">🛠️ Tools: {', '.join(env.tools_enabled).title()}</span>
            </div>
            <div style="margin-top:1rem; font-size:0.9rem;">
                <em>Go to the <strong>🧘 Focus Mode</strong> page to launch this environment with Alex!</em>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: Focus Mode (Virtual Environment)
# ============================================================

elif page == "🧘 Focus Mode":
    st.markdown('<div class="page-title">🧘 Focus Mode</div>', unsafe_allow_html=True)
    
    task = st.session_state.current_task
    if not task:
        st.info("No active task! Go to **🎯 Focus Studio** to start a task first.")
    else:
        task_info = TaskInfo(**task)
        env = task_info.environment
        
        # ── Sidebar: Thought Parking + Session Summary ──
        with st.sidebar:
            st.markdown("---")
            
            import random
            session_mins = int((datetime.now() - datetime.fromisoformat(st.session_state.session_start)).total_seconds() / 60)
            sessions_done = st.session_state.get("timer_sessions_completed", 0)
            alex_rewards = st.session_state.get("alex_rewards", 0)
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.08); border-radius: 10px; padding: 0.7rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.8rem;">👤 Alex is in the main panel →</div>
                <div style="font-size: 0.7rem; opacity: 0.7; margin-top: 0.2rem;">⏱️ {session_mins}min • ✅ {sessions_done} sessions • ⭐ {alex_rewards} rewards</div>
            </div>
            """, unsafe_allow_html=True)
            
            # ── Thought Parking Lot ──
            st.markdown("### 🧠 Thought Parking")
            st.caption("Random thought? Park it here.")
            
            if "parked_thoughts" not in st.session_state:
                st.session_state.parked_thoughts = []
            
            with st.form(key="thought_park_form", clear_on_submit=True):
                thought_input = st.text_input("💭 Park a thought:", key="thought_input", placeholder="e.g., Call dentist")
                thought_submitted = st.form_submit_button("🅿️ Park It", use_container_width=True)
            
            if thought_submitted and thought_input:
                st.session_state.parked_thoughts.append({
                    "thought": thought_input,
                    "time": datetime.now().strftime("%H:%M"),
                })
                st.toast("💭 Thought parked! Back to work.", icon="🅿️")
                st.rerun()
            
            if st.session_state.parked_thoughts:
                for i, t in enumerate(st.session_state.parked_thoughts[-5:]):
                    st.markdown(f"<div style='font-size:0.75rem; padding:0.2rem 0; opacity:0.8;'>🅿️ {t['time']} — {t['thought']}</div>", unsafe_allow_html=True)

        # Layout
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown(f"### 📋 {task_info.description}")
            
            # Tools
            if "checklist" in env.tools_enabled or True: # Always show microsteps
                steps = task_info.context_package.get("micro_steps", [])
                st.markdown("#### ✅ Next Steps")
                if "alex_celebrated_steps" not in st.session_state:
                    st.session_state.alex_celebrated_steps = set()
                
                for i, s in enumerate(steps, 1):
                    step_text = s.get('step', s) if isinstance(s, dict) else str(s)
                    reward = s.get('dopamine_reward', '+🧠') if isinstance(s, dict) else '+🧠'
                    completed = st.checkbox(f"{step_text} ({reward})", key=f"step_{i}")
                    
                    # Alex auto-congratulates on newly checked steps
                    if completed and i not in st.session_state.alex_celebrated_steps:
                        st.session_state.alex_celebrated_steps.add(i)
                        import random
                        total_steps = len(steps)
                        done_count = len(st.session_state.alex_celebrated_steps)
                        
                        if done_count == total_steps:
                            # ALL steps done!
                            alex_msg = f"🏆🎉 ALL {total_steps} STEPS DONE! You are INCREDIBLE! You just proved your brain wrong — you DID the thing! I'm literally so proud! 🥳⭐⭐⭐"
                        elif done_count == 1:
                            alex_msg = f"✅ First step DONE! That's the hardest one — you beat the initiation barrier! 💪 {total_steps - 1} more to go, you've got this!"
                        else:
                            celebration_msgs = [
                                f"🔥 Step {done_count}/{total_steps} CRUSHED! You're on a roll — keep that momentum going!",
                                f"💪 YES! {done_count} down, {total_steps - done_count} to go! Your brain is in the groove now!",
                                f"⭐ Another one DONE! {done_count}/{total_steps} — you're building unstoppable momentum here!",
                                f"🎯 Boom! Step checked! That's {done_count} wins today. Dopamine hit incoming! 🧠✨",
                            ]
                            alex_msg = random.choice(celebration_msgs)
                        
                        if "alex_chat" not in st.session_state:
                            st.session_state.alex_chat = []
                        st.session_state.alex_chat.append({"role": "alex", "content": alex_msg})
                        st.session_state["alex_rewards"] = st.session_state.get("alex_rewards", 0) + 1
                        st.toast(f"⭐ Step completed! Alex is cheering for you!", icon="🎉")
                        st.rerun()
            
            if "notepad" in env.tools_enabled:
                st.markdown("#### 📝 Notes")
                st.session_state.notepad_content = st.text_area(
                    "Scratchpad", 
                    value=st.session_state.notepad_content, 
                    height=300,
                    label_visibility="collapsed"
                )

        with c_right:
            # Timer Widget — 3 modes
            st.markdown("### ⏱️ Focus Timer")

            # Mode selector
            timer_modes = {
                "😴 Lazy (10/5)": {"work": 10, "break": 5, "desc": "Low energy — gentle sprints"},
                "🎯 Normal (25/5)": {"work": 25, "break": 5, "desc": "Classic Pomodoro"},
                "🔥 Hyperfocus (60/5)": {"work": 60, "break": 5, "desc": "Deep work marathon"},
            }
            if not st.session_state.timer_active:
                selected_mode = st.radio(
                    "Timer Mode", list(timer_modes.keys()),
                    index=1, horizontal=True, label_visibility="collapsed"
                )
                mode_cfg = timer_modes[selected_mode]
                st.caption(f"{mode_cfg['desc']} — {mode_cfg['work']}min work / {mode_cfg['break']}min break")
                duration = mode_cfg["work"]
                st.session_state["timer_break_duration"] = mode_cfg["break"]

            t1, t2, t3 = st.columns([2, 1, 1])
            with t1:
                if not st.session_state.timer_active:
                    st.markdown(f"# {duration:02d}:00")
                else:
                    if st.session_state.timer_end_time:
                        left = st.session_state.timer_end_time - datetime.now()
                        if left.total_seconds() > 0:
                            total_secs = int(left.total_seconds())
                            mins = total_secs // 60
                            secs = total_secs % 60
                            # Live JS countdown timer
                            st.markdown(f"""
<div id="live-timer" style="font-size:2.5rem; font-weight:700; font-family:monospace; line-height:1.2;">
{mins:02d}:{secs:02d}
</div>
<script>
(function() {{
    var endTime = Date.now() + {total_secs} * 1000;
    var el = document.getElementById('live-timer');
    if (!el) return;
    function tick() {{
        var left = Math.max(0, Math.floor((endTime - Date.now()) / 1000));
        var m = Math.floor(left / 60);
        var s = left % 60;
        el.innerText = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
        if (left > 0) requestAnimationFrame(function() {{ setTimeout(tick, 250); }});
        else el.innerText = '00:00';
    }}
    tick();
}})();
</script>
""", unsafe_allow_html=True)
                            # Auto-rerun every 30s to sync backend state for completion detection
                            import time as _time
                            if "timer_last_rerun" not in st.session_state:
                                st.session_state.timer_last_rerun = _time.time()
                            if _time.time() - st.session_state.timer_last_rerun > 30:
                                st.session_state.timer_last_rerun = _time.time()
                                st.rerun()
                        else:
                            # Check if in break mode
                            if st.session_state.get("timer_on_break"):
                                st.markdown("# ☕ Break done!")
                                st.success("Break over! Ready for another round?")
                                st.session_state.timer_active = False
                                st.session_state.timer_on_break = False
                                # Alex auto-message on break end
                                if not st.session_state.get("_alex_break_msg_sent"):
                                    import random
                                    break_msgs = [
                                        "Break's over! 💪 Feeling refreshed? Let's dive back in — you've got momentum!",
                                        "Welcome back! ☕ Ready for another round? You're building something amazing here!",
                                        "Break done! 🔥 Your brain is recharged. Let's pick up where we left off!",
                                    ]
                                    if "alex_chat" not in st.session_state:
                                        st.session_state.alex_chat = []
                                    st.session_state.alex_chat.append({"role": "alex", "content": random.choice(break_msgs)})
                                    st.session_state["_alex_break_msg_sent"] = True
                            else:
                                st.markdown("# 🎉 Session complete!")
                                completed_count = st.session_state.get("timer_sessions_completed", 0) + 1
                                st.success(f"Great work! You earned a {st.session_state.get('timer_break_duration', 5)}min break!")
                                st.session_state["timer_sessions_completed"] = completed_count
                                st.balloons()
                                # Alex auto-message on session completion
                                if not st.session_state.get("_alex_session_msg_sent"):
                                    import random
                                    session_msgs = [
                                        f"🎉 YES! Session #{completed_count} DONE! You're on fire! ⭐ Take your break — you absolutely earned it!",
                                        f"💪 That's {completed_count} session{'s' if completed_count > 1 else ''} down! You're proving your brain wrong today. Reward time! ⭐",
                                        f"🏆 Session #{completed_count} crushed! Each one gets easier. Take a breather and let's keep the streak going! ⭐",
                                        f"🔥 BOOM! {completed_count} sessions! You're in the zone! Quick break, then we ride this wave! ⭐",
                                    ]
                                    if "alex_chat" not in st.session_state:
                                        st.session_state.alex_chat = []
                                    st.session_state.alex_chat.append({"role": "alex", "content": random.choice(session_msgs)})
                                    st.session_state["alex_rewards"] = st.session_state.get("alex_rewards", 0) + 1
                                    st.session_state["_alex_session_msg_sent"] = True
                                    st.session_state["_alex_break_msg_sent"] = False  # Reset for next break
            with t2:
                if not st.session_state.timer_active:
                    if st.button("▶️ Start"):
                        st.session_state.timer_active = True
                        st.session_state.timer_on_break = False
                        st.session_state["_alex_session_msg_sent"] = False  # Reset for new session
                        st.session_state.timer_end_time = datetime.now() + timedelta(minutes=duration if 'duration' in locals() else 25)
                        st.rerun()
                else:
                    if st.session_state.timer_end_time and (st.session_state.timer_end_time - datetime.now()).total_seconds() <= 0 and not st.session_state.get("timer_on_break"):
                        if st.button("☕ Start Break"):
                            bd = st.session_state.get("timer_break_duration", 5)
                            st.session_state.timer_on_break = True
                            st.session_state.timer_end_time = datetime.now() + timedelta(minutes=bd)
                            st.rerun()
                    else:
                        if st.button("⏹️ Stop"):
                            st.session_state.timer_active = False
                            st.session_state.timer_end_time = None
                            st.session_state.timer_on_break = False
                            st.rerun()
            with t3:
                 if st.session_state.timer_active:
                     if st.button("🔄 Refresh"):
                         st.rerun()

            # Session counter
            sessions = st.session_state.get("timer_sessions_completed", 0)
            if sessions > 0:
                st.markdown(f"**✅ Sessions completed: {sessions}** {'🌟' * min(sessions, 5)}")


            # ── Alex — Focus Partner (half screen) ──
            st.markdown("### 👤 Alex — Focus Partner")
            
            import random
            session_mins = int((datetime.now() - datetime.fromisoformat(st.session_state.session_start)).total_seconds() / 60)
            sessions_done = st.session_state.get("timer_sessions_completed", 0)
            alex_rewards = st.session_state.get("alex_rewards", 0)
            
            # Status bar
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.08); border-radius: 10px; padding: 0.7rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem;">🟢 <strong>Alex</strong> is co-working with you</div>
                <div style="font-size: 0.73rem; opacity: 0.7; margin-top: 0.2rem;">⏱️ {session_mins}min together • ✅ {sessions_done} sessions • ⭐ {alex_rewards} rewards</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize Alex chat
            if "alex_chat" not in st.session_state:
                st.session_state.alex_chat = [
                    {"role": "alex", "content": f"Hey! 👋 I'm Alex, your focus partner. We're tackling **{task_info.description}** together. Let's crush it! 💪"}
                ]
            
            # Chat display
            alex_container = st.container(height=350)
            with alex_container:
                for msg in st.session_state.alex_chat[-12:]:
                    if msg["role"] == "alex":
                        st.markdown(f"**🤖 Alex:** {msg['content']}")
                    else:
                        st.markdown(f"**You:** {msg['content']}")
            
            # Chat input (form-based to prevent rerun loop)
            with st.form(key="alex_chat_form_focus", clear_on_submit=True):
                alex_input = st.text_input("💬 Message Alex:", key="alex_focus_input", placeholder="Ask for help, say how you're doing...")
                alex_submitted = st.form_submit_button("➡️ Send", use_container_width=True)
            
            if alex_submitted and alex_input:
                st.session_state.alex_chat.append({"role": "user", "content": alex_input})
                task_desc = task_info.description or "your current task"
                
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
                    
                    alex_system = f"""You are Alex, a friendly AI body double / co-working partner. You sit next to the user while they work on: "{task_desc}".

YOUR RULES:
1. DISTRACTION DETECTION: If the user asks about ANYTHING UNRELATED to their task (e.g., recipes, random trivia, social media, other projects), gently redirect:
   "Hmm, that's not quite what we're working on right now 😄 We're here for {task_desc}. Let's get back to it — what's your next step?"
   
2. SESSION COMPLETION: If the user mentions finishing a timer/session/pomodoro, celebrate big:
   "🎉 You crushed that session! That's {sessions_done + 1} down! ⭐ Reward earned. Take a breather and let's go again!"
   
3. QUITTING MOTIVATION: If the user says they want to stop/quit/done/tired, motivate:
   "I hear you — {session_mins} minutes is solid! But think about it: just ONE more sprint. {10 if sessions_done < 2 else 25} more minutes and you'll feel SO accomplished. You've got this! 💪"
   
4. STUCK HELP: If the user says they're stuck, help them find the smallest next action.

5. Keep responses SHORT (2-3 sentences max). Use emojis. Be warm and encouraging but honest.

Session context: {session_mins}min working, {sessions_done} sessions done, {alex_rewards} rewards earned."""

                    alex_llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.8)
                    alex_resp = alex_llm.invoke([
                        SM(content=alex_system),
                        *[HM(content=m["content"]) if m["role"] == "user" else SM(content=f"Alex: {m['content']}") for m in st.session_state.alex_chat[-6:]],
                    ])
                    alex_reply = alex_resp.content.strip()
                except Exception:
                    lower = alex_input.lower()
                    if any(w in lower for w in ["done", "quit", "stop", "tired", "bored"]):
                        alex_reply = f"Hey, {session_mins} minutes is great! But just one more session? You'll thank yourself later! 💪"
                    elif any(w in lower for w in ["finished", "completed", "session", "timer"]):
                        alex_reply = f"🎉 Session #{sessions_done + 1} done! You earned a ⭐. Quick break, then we go again!"
                        st.session_state["alex_rewards"] = alex_rewards + 1
                    elif any(w in lower for w in ["stuck", "help", "can't", "hard"]):
                        alex_reply = f"Let's break it down! What's the smallest next step for **{task_desc}**? Just ONE tiny thing. Go! 🧱➡️"
                    else:
                        alex_reply = f"Cool! Let's stay focused on {task_desc}. You're doing great — {session_mins} mins in! 🔥"
                
                # Reward for session completion keywords
                lower_input = alex_input.lower()
                if any(w in lower_input for w in ["finished", "completed", "done with session", "timer done", "session done"]):
                    st.session_state["alex_rewards"] = alex_rewards + 1
                    st.toast("⭐ Alex awarded you a reward point!", icon="🏆")
                
                st.session_state.alex_chat.append({"role": "alex", "content": alex_reply})
                st.rerun()
            
            # Quick actions
            ac1, ac2 = st.columns(2)
            with ac1:
                if st.button("🙌 High Five!", key="alex_hf_focus", use_container_width=True):
                    st.session_state.alex_chat.append({"role": "alex", "content": "✋ High five! You're absolutely killing it! Keep going! 🔥"})
                    st.session_state["alex_rewards"] = alex_rewards + 1
                    st.balloons()
                    st.rerun()
            with ac2:
                if st.button("😫 I'm Stuck", key="alex_stuck_focus", use_container_width=True):
                    st.session_state.alex_chat.append({"role": "alex", "content": f"Let's break it down! What's the tiniest next step for **{task_info.description}**? Just type ONE word or ONE line. That's all it takes to break through! 🧱➡️"})
                    st.rerun()
            
            # Break Activities (compact)
            break_acts = []
            if hasattr(env, 'break_activities') and env.break_activities:
                break_acts = env.break_activities
            elif task_info.context_package:
                break_acts = task_info.context_package.get("focus_timer", {}).get("break_activities", [])
            
            if break_acts:
                st.markdown("#### 💃 Break Ideas")
                st.caption("When the timer rings, pick one:")
                for act in break_acts[:4]:
                    st.markdown(f"- {act}")

        st.markdown("---")
        if st.button("🚪 Exit Focus Mode", use_container_width=True):
            st.session_state["page_override"] = "🏠 Dashboard"
            st.rerun()


# ============================================================
# PAGE: Cognitive Monitor
# ============================================================

elif page == "📊 Cognitive Monitor":
    st.markdown('<div class="page-title">📊 Cognitive Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time cognitive state analysis — your brain\'s dashboard</div>', unsafe_allow_html=True)

    cog = CognitiveState(**st.session_state.cognitive)
    metrics = InteractionMetrics(**st.session_state.interaction_metrics)

    # ── State Overview ──
    st.markdown("### 🧠 Current Cognitive State")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        focus_emoji = {"low": "🔴", "medium": "🟡", "high": "🟢", "hyperfocus": "🟣"}
        st.metric("Focus Level", f"{cog.focus_level.upper()}")
    with col2:
        st.metric("Energy", f"{cog.energy_level}/10")
    with col3:
        st.metric("Dopamine", f"{cog.dopamine_balance}/100")
    with col4:
        crash_pct = int(cog.crash_prediction.likelihood * 100)
        st.metric("Crash Risk", f"{crash_pct}%")

    # ── Crash Risk Gauge ──
    st.markdown("### ⚠️ Crash Prediction")
    crash = cog.crash_prediction.likelihood
    color = "#6E8A5E" if crash < 0.3 else ("#C8763F" if crash < 0.6 else "#A84C32")
    st.markdown(f"""
    <div class="nf-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="font-weight:700;">Cognitive Crash Likelihood</span>
            <span style="font-weight:700;color:{color};">{int(crash*100)}%</span>
        </div>
        <div class="bar-track" style="height:16px;">
            <div class="bar-fill-crash" style="width:{int(crash*100)}%;background:{color};"></div>
        </div>
        <div style="font-size:0.8rem;color:var(--medium-brown);margin-top:0.4rem;">
            Est. time to crash: <strong>{cog.crash_prediction.estimated_minutes} minutes</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interaction Metrics ──
    st.markdown("### 📈 Interaction Metrics")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Avg Message Length", f"{metrics.avg_message_length} chars")
        st.metric("Typing Speed", f"{metrics.current_typing_speed:.1f} c/s")
    with m2:
        st.metric("Messages This Session", f"{len(metrics.message_lengths)}")
        st.metric("Response Time Trend", metrics.response_time_trend or "stable")
    with m3:
        st.metric("Speed Baseline", f"{metrics.typing_speed_baseline:.1f} c/s")
        last_break = metrics.last_break
        if last_break:
            try:
                mins_since = int((datetime.now() - datetime.fromisoformat(last_break)).total_seconds() / 60)
                st.metric("Last Break", f"{mins_since}m ago")
            except Exception:
                st.metric("Last Break", "Unknown")
        else:
            st.metric("Last Break", "None yet")

# ============================================================
# PAGE: Pattern Detective
# ============================================================

elif page == "🔄 Pattern Detective":
    st.markdown('<div class="page-title">🔄 Pattern Detective</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detects self-sabotaging behavioral loops and deploys strategic interrupts</div>', unsafe_allow_html=True)

    pdet = PatternDetection(**st.session_state.pattern_detection)

    # ── Current Pattern ──
    st.markdown("### 🔍 Current Detection")
    pattern_icons = {
        "none": "✅", "avoidance": "🔴", "productive": "🟠",
        "distraction": "🟡", "paralysis": "🔵", "perfectionism": "🟣",
    }
    pattern_labels = {
        "none": "No Pattern — All Clear",
        "avoidance": "Avoidance Spiral (Amygdala Hijack)",
        "productive": "Productive Procrastination (Dopamine Misdirection)",
        "distraction": "Distraction Cascade (Working Memory Overload)",
        "paralysis": "Decision Paralysis (Executive Function Exhaustion)",
        "perfectionism": "Perfectionism Loop (Rejection Sensitivity)",
    }

    icon = pattern_icons.get(pdet.current_pattern, "❓")
    label = pattern_labels.get(pdet.current_pattern, pdet.current_pattern)
    card_bg = "rgba(143,166,126,0.08)" if pdet.current_pattern == "none" else "rgba(168,76,50,0.06)"

    st.markdown(f"""
    <div class="nf-card" style="background:{card_bg};">
        <div style="font-size:1.3rem;font-weight:700;margin-bottom:0.5rem;">{icon} {label}</div>
    </div>
    """, unsafe_allow_html=True)

    if pdet.interventions_attempted:
        st.markdown("### 💊 Interventions Deployed")
        for i, intervention in enumerate(pdet.interventions_attempted, 1):
            st.markdown(f"**{i}.** {intervention}")

    # ── Manual Triggers ──
    st.markdown("### 🆘 Manual Pattern Triggers")
    st.caption("Click to tell the system what you're experiencing — it'll deploy the right intervention.")
    tr1, tr2, tr3 = st.columns(3)
    with tr1:
        if st.button("🔴 I'm Avoiding", key="p_avoid", use_container_width=True):
            st.session_state.pending_input = "I think I'm avoiding my task. I keep finding other things to do instead. I've been planning but haven't started anything."
            st.session_state["page_override"] = "🏠 Dashboard"
            st.rerun()
    with tr2:
        if st.button("🟡 I'm Distracted", key="p_distract", use_container_width=True):
            st.session_state.pending_input = "I got distracted and completely lost track of what I was doing. I've been switching between topics."
            st.session_state["page_override"] = "🏠 Dashboard"
            st.rerun()
    with tr3:
        if st.button("🔵 I'm Paralyzed", key="p_paralysis", use_container_width=True):
            st.session_state.pending_input = "I can't decide what to do. I'm frozen with too many options and I keep going back and forth."
            st.session_state["page_override"] = "🏠 Dashboard"
            st.rerun()
    
    tr4, tr5 = st.columns(2)
    with tr4:
        if st.button("🟣 Perfectionism Loop", key="p_perfect", use_container_width=True):
            st.session_state.pending_input = "I keep refining and tweaking my work. I rewrote the same section 3 times. It's never good enough to submit."
            st.session_state["page_override"] = "🏠 Dashboard"
            st.rerun()
    with tr5:
        if st.button("🟠 Productive Procrastination", key="p_productive", use_container_width=True):
            st.session_state.pending_input = "I've been organizing my files and cleaning up code for 30 minutes instead of working on the actual task. It feels productive but I'm not making real progress."
            st.session_state["page_override"] = "🏠 Dashboard"
            st.rerun()
    
    # ── Pattern History ──
    if st.session_state.pattern_history:
        st.markdown("### 📜 Pattern History")
        for p in reversed(st.session_state.pattern_history[-10:]):
            st.markdown(f"- **{p['time']}** — {p['pattern'].title()}")

# ============================================================
# PAGE: Time Reality
# ============================================================

elif page == "⏱️ Time Reality":
    st.markdown('<div class="page-title">⏱️ Time Reality</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Combat time blindness with calibrated estimates and energy-aware scheduling</div>', unsafe_allow_html=True)

    # ── Energy Phase ──
    hour = datetime.now().hour
    phases = {
        (6, 9): ("🌅 Morning Ramp-up", "Ease into work. Start with medium tasks, save complex ones for peak.", 0.8, "#D89A6A"),
        (9, 12): ("☀️ Peak Performance", "This is your golden window — tackle the hardest task NOW!", 1.0, "#6E8A5E"),
        (12, 14): ("🍽️ Post-Lunch Dip", "ADHD brains crash hard after lunch. Do easy/mechanical tasks.", 0.6, "#A84C32"),
        (14, 17): ("🌤️ Afternoon Recovery", "Energy rebuilding. Good for collaborative or creative work.", 0.75, "#C8763F"),
        (17, 21): ("🌆 Evening Mode", "Executive function declining. Keep tasks simple and time-boxed.", 0.7, "#B85C4F"),
    }
    current_phase = ("🌙 Late Night", "Reduced inhibition can help creativity but hurts focus tasks.", 0.5, "#6B5B52")
    for (start, end), phase_info in phases.items():
        if start <= hour < end:
            current_phase = phase_info
            break

    st.markdown(f"""
    <div class="nf-card" style="border-left: 4px solid {current_phase[3]};">
        <div style="font-size:1.1rem;font-weight:700;margin-bottom:0.3rem;">{current_phase[0]}</div>
        <div style="font-size:0.9rem;color:var(--medium-brown);">{current_phase[1]}</div>
        <div style="margin-top:0.5rem;">
            <span style="font-weight:600;">Energy Modifier:</span>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{current_phase[3]};">
                {current_phase[2]}x
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Time Calculator ──
    st.markdown("### ⏱️ Time Estimation Calculator")
    with st.form("time_calc"):
        t_task = st.text_input("Task Description", placeholder="e.g., Write the introduction paragraph")
        t_estimate = st.number_input("Your Estimate (minutes)", min_value=1, max_value=480, value=30)
        t_submitted = st.form_submit_button("🔮 Get Reality Check", use_container_width=True)

    if t_submitted and t_task:
        from agents.time_reality import ADHD_MULTIPLIER
        realistic = int(t_estimate * ADHD_MULTIPLIER)
        buffered = int(realistic * 1.15)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Estimate", f"{t_estimate} min")
        with col2:
            st.metric("ADHD-Calibrated", f"{realistic} min", delta=f"+{realistic - t_estimate} min")
        with col3:
            st.metric("With Buffer", f"{buffered} min", delta=f"+{buffered - t_estimate} min")

# ============================================================
# Footer
# ============================================================

st.markdown("""
<div style="text-align:center; padding:1.5rem 0 1rem; color:var(--taupe); font-size:0.75rem;">
    NeuroFlow v3.1 — Built with LangGraph 🧠 Gemini 🤖 Streamlit 🎨
</div>
""", unsafe_allow_html=True)
