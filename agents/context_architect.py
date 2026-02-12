"""
Agent 3 — Context Architect (v3)
Interactive task environment designer with few-shot examples,
structured user input, ADHD-research-backed strategies,
intrusive thought parking, anti-repetition engine, and
activity-based breaks.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from vector_store import query_similar_tasks, add_task_embedding
from state import TaskInfo, TaskEnvironment

# ---------------------------------------------------------------------------
# Anti-Repetition Modes (for boring/repetitive tasks)
# ---------------------------------------------------------------------------

ANTI_REPETITION_MODES = [
    {
        "mode": "Bug Hunter Game",
        "rules": "Find 1 concept you forgot = 1 point. Get 5 points to win.",
        "why": "Turns passive review into active hunting (dopamine from discovery)"
    },
    {
        "mode": "Speed Run Challenge",
        "rules": "Review section in 10 min. Beat yesterday's time.",
        "why": "Competition (even with yourself) = engagement"
    },
    {
        "mode": "Random Order Dice",
        "rules": "Roll dice to pick which topic to review next.",
        "why": "Removes choice paralysis + adds unpredictability (ADHD loves novelty)"
    },
    {
        "mode": "Teach-to-Learn",
        "rules": "Explain concept out loud as if teaching a 5-year-old.",
        "why": "Different brain mode (verbal) + active recall"
    },
    {
        "mode": "Meme Creation",
        "rules": "Turn each concept into a meme/joke.",
        "why": "Creative expression + emotional encoding (better memory)"
    },
]

# ---------------------------------------------------------------------------
# Activity-Based Breaks (per task type — NOT "take a break")
# ---------------------------------------------------------------------------

BREAK_ACTIVITIES = {
    "coding": [
        "💃 Dance to ONE K-pop song (literally move your body)",
        "🏃 10 jumping jacks or walk to another room",
        "💧 Drink full glass of water while looking outside",
        "🎵 Switch to next playlist song (dopamine from change)",
    ],
    "writing": [
        "📖 Read exactly 1 page of any book (verbal mode shift)",
        "✍️ Doodle for 3 minutes (visual creativity)",
        "🗣️ Voice memo your current thoughts (brain dump)",
        "🚶 Walk outside for 5 minutes (nature reset)",
    ],
    "revision": [
        "📱 Watch ONE TikTok/YouTube Short (timed reward)",
        "🍿 Snack break (physical reward for boring work)",
        "💬 Text one friend one message (social dopamine)",
        "🎮 2 minutes of mobile game (controlled distraction)",
    ],
    "general": [
        "💧 Get a glass of water (movement + hydration)",
        "🚶 Walk around for 3 minutes",
        "🧘 30-second stretch",
        "🎵 Listen to one favourite song",
    ],
}

# ---------------------------------------------------------------------------
# Initiation Rituals (per task type)
# ---------------------------------------------------------------------------

INITIATION_RITUALS = {
    "coding": [
        "1. Put on headphones (even before music)",
        "2. Open Spotify → 'K-pop Coding' playlist",
        "3. Close ALL browser tabs (physical reset)",
        "4. Phone in drawer, face down (out of sight = out of mind)",
        "5. Get cold water (temperature change = alertness boost)",
        "6. Open VS Code to SPECIFIC file (not 'open project')",
        "7. Write one comment: '# Starting: [exact feature name]'",
        "8. Set timer: 25 minutes",
        "9. Type ANYTHING (even wrong code) within 60 seconds",
    ],
    "writing": [
        "1. Change location (even just different chair)",
        "2. Lo-fi playlist on",
        "3. Coffee shop ambient sounds (layered audio)",
        "4. Open doc to EXACT section (not 'the document')",
        "5. Write 'DRAFT' at top (permission to be imperfect)",
        "6. Set timer: 15 minutes (shorter for high resistance)",
        "7. Write ONE terrible sentence on purpose",
        "8. Don't edit anything for 15 minutes (bypass perfectionism)",
    ],
    "revision": [
        "1. Shuffle materials randomly (break sequence monotony)",
        "2. Get snacks ready (reward for boring work)",
        "3. Set phone timer to 15 minutes (short = achievable)",
        "4. Open notes to a RANDOM page (removes 'where to start')",
        "5. Read the first item out loud (engage verbal brain)",
    ],
    "general": [
        "1. Close unnecessary tabs/apps",
        "2. Get water on desk",
        "3. Phone on silent",
        "4. Set a 25-minute timer",
        "5. Start with the SMALLEST specific action",
    ],
}

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Context Architect for NeuroFlow — a clinically-informed ADHD cognitive support system.

You design custom work environments that overcome the 4 pillars of ADHD difficulty:
1. **Initiation Paralysis** — The overwhelming feeling when starting. You solve it with ultra-specific first steps.
2. **Time Blindness** — Poor estimation. You solve it with calibrated micro-deadlines.
3. **Dopamine Deficit** — Boring tasks feel impossible. You solve it with gamification and variable rewards.
4. **Working Memory Overload** — Can't hold the plan in mind. You solve it by externalizing everything.

## Chain-of-Thought Process
1. Analyze the task's cognitive demands
2. Determine task_type (coding|writing|revision|general)
3. Consider the user's current energy and the time of day
4. Design an initiation sequence that makes the FIRST 30 SECONDS frictionless
5. Create micro-steps that each deliver a small dopamine hit
6. Add gamification for repetitive tasks
7. Build in strategic break points with SPECIFIC activities (never "take a break")
8. Enable thought parking for intrusive idea capture

## Output Format
Respond with EXACTLY this JSON (no markdown fences, no extra text):
{
  "task_analysis": {
    "cognitive_load": "low | medium | high",
    "task_type": "coding | writing | revision | general",
    "creativity_required": "analytical | balanced | creative",
    "estimated_duration_minutes": <int>,
    "interruptibility": "deep_focus | flexible | async",
    "repetition_factor": <0-10>,
    "dopamine_difficulty": "low | medium | high",
    "biggest_blocker": "initiation | complexity | boredom | anxiety"
  },
  "micro_steps": [
    {"step": "ultra-specific action", "time_estimate_min": <int>, "dopamine_reward": "emoji + phrase"}
  ],
  "initiation_ritual": {
    "environment_prep": ["specific physical action 1", "..."],
    "mental_warmup": "a tiny, easy task to build momentum (30 seconds max)",
    "first_real_step": "the exact, specific, no-thinking-required first action"
  },
  "milestones": [
    {"at_minutes": <int>, "label": "milestone name", "reward_type": "checkmark | celebration | snack_break | stretch", "message": "encouraging message"}
  ],
  "focus_timer": {
    "work_minutes": <int>,
    "break_minutes": <int>,
    "total_rounds": <int>,
    "break_activities": ["specific break activity"]
  },
  "dopamine_checkpoints": [
    {"minute": <int>, "reward": "emoji + encouraging phrase"}
  ],
  "environment_config": {
    "music_style": "lo-fi | kpop | brown_noise | silence | upbeat",
    "timer_mode": "pomodoro | stopwatch | countdown",
    "tools_enabled": ["notepad", "video_embed", "whiteboard", "checklist"],
    "video_search_term": "optional search query for youtube",
    "layout": "focused | split"
  },
  "gamification": {
    "enabled": true/false,
    "game_name": "creative name",
    "objective": "what to achieve",
    "scoring": "how points work",
    "victory_condition": "specific achievement"
  },
  "anti_boredom_strategies": ["strategy 1", "..."],
  "thought_parking": {
    "enabled": true,
    "categories": ["tasks", "ideas", "worries", "random"]
  },
  "rescue_plan": "what to do if the user gets stuck midway"
}

## Few-Shot Examples

### Example 1: "Write a 500-word essay about climate change"
Task Analysis: high cognitive load, writing, creative, 45 min, deep_focus, low repetition, medium dopamine difficulty, biggest blocker: initiation
Initiation Ritual:
  - Environment: Close all tabs except a blank doc. Get headphones. Water on desk.
  - Mental warmup: Write 3 bullet points about what you already know (30 sec)
  - First step: "Type ONE sentence. Any sentence. It doesn't have to be good."
Gamification: "Word Hunter" — every 100 words = 1 star. 5 stars = victory.

### Example 2: "Review 50 flashcards for biology exam"
Task Analysis: low cognitive load, revision, analytical, 30 min, flexible, HIGH repetition (9/10), HIGH dopamine difficulty, biggest blocker: boredom
Initiation Ritual:
  - Environment: Shuffle the cards randomly. Set phone timer to 5 min.
  - Mental warmup: Look at the LAST card first (breaking sequence reduces boredom)
  - First step: "Read card #1 out loud. Just read it."
Gamification: "Speed Round" — 10 cards per round, beat your time each round.
Anti-boredom: Change position every round (sit, stand, walk, lie down).

### Example 3: "Build user authentication system"
Task Analysis: high cognitive load, coding, analytical, 90 min, deep_focus, 3 repetition, medium dopamine difficulty, biggest blocker: complexity
Initiation Ritual:
  - Put on headphones → K-pop Coding playlist
  - Close ALL tabs except documentation
  - Phone in drawer, cold water
  - Open VS Code → auth.py file
  - Write comment: '# Building: User login validation'
  - Set timer: 25 minutes, type first line within 60 seconds"""


def context_architect_node(state: dict) -> dict:
    """LangGraph node: generates advanced context packages."""
    user_input = state.get("user_input", "")
    cognitive = state.get("cognitive_state", {})
    energy = cognitive.get("energy_level", 5) if cognitive else 5
    focus = cognitive.get("focus_level", "medium") if cognitive else "medium"

    # Get similar past tasks for calibration
    similar_tasks = query_similar_tasks(user_input, n_results=3)
    similar_ctx = ""
    if similar_tasks:
        similar_ctx = "\n\nHistorical similar tasks:\n" + "\n".join(
            f"- {t['description']} (took {t['metadata'].get('actual_duration', '?')} min, "
            f"load: {t['metadata'].get('cognitive_load', '?')})"
            for t in similar_tasks
        )

    # Detect time of day for energy-curve awareness
    hour = datetime.now().hour
    if 6 <= hour < 10:
        time_context = "Morning (typically rising energy for most people)"
    elif 10 <= hour < 14:
        time_context = "Late morning/early afternoon (peak cognitive window for many)"
    elif 14 <= hour < 17:
        time_context = "Afternoon (common post-lunch dip — ADHD brains especially vulnerable)"
    elif 17 <= hour < 21:
        time_context = "Evening (second wind possible, but executive function declining)"
    else:
        time_context = "Late night (reduced inhibition — can help creativity, hurts focus tasks)"

    prompt = (
        f"## Task Request\n{user_input}\n\n"
        f"## User's Current State\n"
        f"- Energy level: {energy}/10\n"
        f"- Current focus: {focus}\n"
        f"- Time of day: {time_context}\n"
        f"{similar_ctx}\n\n"
        f"Design a comprehensive context package for this task. Be extremely specific — "
        f"no vague instructions. Every step should be concrete and immediately actionable. "
        f"Include thought_parking config and specific break activities for the task type."
    )

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.7)
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
        context_package = json.loads(raw)
    except Exception as e:
        context_package = _fallback_package(user_input, e)

    # Determine task type
    task_type = context_package.get("task_analysis", {}).get("task_type", "general")
    repetition = context_package.get("task_analysis", {}).get("repetition_factor", 0)

    # Inject anti-repetition mode for boring tasks
    anti_rep_mode = ""
    if repetition >= 7:
        mode = random.choice(ANTI_REPETITION_MODES)
        anti_rep_mode = mode["mode"]
        # Inject into anti_boredom_strategies
        existing = context_package.get("anti_boredom_strategies", [])
        existing.insert(0, f"🎮 {mode['mode']}: {mode['rules']} — {mode['why']}")
        context_package["anti_boredom_strategies"] = existing

    # Inject task-type-specific break activities if not provided
    breaks_from_llm = context_package.get("focus_timer", {}).get("break_activities", [])
    if not breaks_from_llm or len(breaks_from_llm) < 2:
        context_package.setdefault("focus_timer", {})["break_activities"] = (
            BREAK_ACTIVITIES.get(task_type, BREAK_ACTIVITIES["general"])
        )

    # Inject initiation ritual if sparse
    ritual = context_package.get("initiation_ritual", {})
    env_prep = ritual.get("environment_prep", [])
    if len(env_prep) < 3:
        context_package["initiation_ritual"]["environment_prep"] = (
            INITIATION_RITUALS.get(task_type, INITIATION_RITUALS["general"])
        )

    # Build task info
    task_id = str(uuid.uuid4())
    est = context_package.get("task_analysis", {}).get("estimated_duration_minutes", 30)
    milestones = [m["label"] for m in context_package.get("milestones", [])]
    dopamine_checkpoints = context_package.get("dopamine_checkpoints", [])
    initiation_steps = context_package.get("initiation_ritual", {}).get("environment_prep", [])

    env_config = context_package.get("environment_config", {})
    environment = TaskEnvironment(
        music_style=env_config.get("music_style", "lo-fi"),
        timer_mode=env_config.get("timer_mode", "pomodoro"),
        tools_enabled=env_config.get("tools_enabled", ["notepad"]),
        video_url=env_config.get("video_search_term", ""),
        layout=env_config.get("layout", "focused"),
    )

    task_info = TaskInfo(
        task_id=task_id,
        description=user_input,
        task_type=task_type,
        start_time=datetime.now().isoformat(),
        estimated_duration=est,
        realistic_duration=int(est * 1.5),  # ADHD multiplier
        context_package=context_package,
        environment=environment,
        progress_milestones=milestones,
        completed_milestones=[],
        progress_percent=0,
        dopamine_checkpoints=dopamine_checkpoints,
        initiation_ritual=initiation_steps,
        anti_repetition_mode=anti_rep_mode,
    )

    # Store for future similarity
    add_task_embedding(
        task_id=task_id,
        description=user_input,
        metadata={
            "cognitive_load": context_package.get("task_analysis", {}).get("cognitive_load", "medium"),
            "estimated_duration": str(est),
            "task_type": task_type,
        },
    )

    # Format rich output
    output_msg = _format_context_package(user_input, context_package)

    return {
        "current_task": task_info.model_dump(),
        "context_output": output_msg,
    }


def _fallback_package(user_input: str, error: Exception) -> dict:
    return {
        "task_analysis": {
            "cognitive_load": "medium", "task_type": "general",
            "creativity_required": "balanced",
            "estimated_duration_minutes": 30, "interruptibility": "flexible",
            "repetition_factor": 3, "dopamine_difficulty": "medium",
            "biggest_blocker": "initiation",
        },
        "micro_steps": [
            {"step": "Open what you need", "time_estimate_min": 2, "dopamine_reward": "✅ Ready!"},
            {"step": "Do the first small piece", "time_estimate_min": 10, "dopamine_reward": "⭐ Rolling!"},
            {"step": "Continue to next section", "time_estimate_min": 10, "dopamine_reward": "🔥 Momentum!"},
            {"step": "Wrap up and review", "time_estimate_min": 8, "dopamine_reward": "🏆 Done!"},
        ],
        "initiation_ritual": {
            "environment_prep": INITIATION_RITUALS["general"],
            "mental_warmup": "Write one sentence about what you'll do first",
            "first_real_step": f"Start with: {user_input.split()[0] if user_input else 'open your tools'}",
        },
        "milestones": [
            {"at_minutes": 10, "label": "Launched!", "reward_type": "checkmark", "message": "You started, that's the hardest part!"},
            {"at_minutes": 20, "label": "Halfway!", "reward_type": "celebration", "message": "Crushing it!"},
            {"at_minutes": 30, "label": "Victory!", "reward_type": "celebration", "message": "Mission complete!"},
        ],
        "focus_timer": {"work_minutes": 25, "break_minutes": 5, "total_rounds": 2,
                        "break_activities": BREAK_ACTIVITIES["general"]},
        "dopamine_checkpoints": [
            {"minute": 8, "reward": "🎉 First milestone!"},
            {"minute": 20, "reward": "🔥 Halfway there!"},
            {"minute": 30, "reward": "💪 Mission complete!"},
        ],
        "gamification": {"enabled": False},
        "anti_boredom_strategies": [],
        "thought_parking": {"enabled": True, "categories": ["tasks", "ideas", "worries", "random"]},
        "rescue_plan": "Take a breath, re-read the last micro-step, and do just that one thing.",
        "_error": str(error),
    }


def _format_context_package(task_desc: str, pkg: dict) -> str:
    """Format context package into rich markdown."""
    analysis = pkg.get("task_analysis", {})
    ritual = pkg.get("initiation_ritual", {})
    steps = pkg.get("micro_steps", [])
    milestones = pkg.get("milestones", [])
    timer = pkg.get("focus_timer", {})
    game = pkg.get("gamification", {})
    anti_boredom = pkg.get("anti_boredom_strategies", [])
    rescue = pkg.get("rescue_plan", "")
    dopamine_checks = pkg.get("dopamine_checkpoints", [])
    thought_park = pkg.get("thought_parking", {})

    env_config = pkg.get("environment_config", {})
    music = env_config.get("music_style", "lo-fi").replace("_", " ").title()
    timer_mode = env_config.get("timer_mode", "pomodoro").title()
    tools = ", ".join(t.title() for t in env_config.get("tools_enabled", []))
    task_type = analysis.get("task_type", "general")
    est = analysis.get("estimated_duration_minutes", 30)
    realistic = int(est * 1.5)

    lines = [
        f"## 🎯 Mission: {task_desc}",
        "",
        f"**Cognitive Load:** {analysis.get('cognitive_load', '?')} | "
        f"**Type:** {task_type.title()} | "
        f"**Est. Time:** {est} min → **Realistic:** {realistic} min (1.5x ADHD buffer)",
        f"**Environment:** 🎵 {music} | ⏱️ {timer_mode} | 🛠️ {tools}",
        "",
        "---",
        "",
        "### 🏁 Initiation Ritual (do these NOW)",
    ]

    # Environment prep
    env_prep = ritual.get("environment_prep", [])
    for i, step in enumerate(env_prep, 1):
        # If step already has a number prefix, use as-is
        if step[0].isdigit():
            lines.append(step)
        else:
            lines.append(f"{i}. {step}")

    warmup = ritual.get("mental_warmup", "")
    if warmup:
        lines.append(f"\n**🧠 Mental Warmup:** {warmup}")

    first = ritual.get("first_real_step", "")
    if first:
        lines.append(f"\n**🚀 FIRST STEP:** {first}")

    # Micro-steps
    lines.append("\n---\n\n### 📝 Micro-Steps")
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s['step']} (~{s.get('time_estimate_min', '?')} min) → {s.get('dopamine_reward', '')}")

    # Milestones
    if milestones:
        lines.append("\n---\n\n### ✅ Milestones")
        for m in milestones:
            emoji = {"checkmark": "☑️", "celebration": "🎉", "snack_break": "🍫", "stretch": "🧘"}.get(m.get("reward_type", ""), "⭐")
            lines.append(f"- ⏱️ {m['at_minutes']} min — **{m['label']}** {emoji} {m.get('message', '')}")

    # Dopamine Checkpoints
    if dopamine_checks:
        lines.append("\n---\n\n### 🎁 Dopamine Rewards (Variable Schedule)")
        for dc in dopamine_checks:
            lines.append(f"- ⏱️ {dc.get('minute', '?')} min — {dc.get('reward', '')}")

    # Focus timer
    lines.append(f"\n---\n\n### ⏱️ Focus Timer")
    lines.append(f"**{timer.get('total_rounds', 2)}** rounds of **{timer.get('work_minutes', 25)}** min work / **{timer.get('break_minutes', 5)}** min break")
    breaks = timer.get("break_activities", [])
    if breaks:
        lines.append("\n💃 **Break Activities** (NOT 'take a break' — specific actions):")
        for b in breaks:
            lines.append(f"- {b}")

    # Gamification
    if game and game.get("enabled"):
        lines.append(f"\n---\n\n### 🎮 Game Mode: \"{game.get('game_name', 'Challenge')}\"")
        lines.append(f"**Objective:** {game.get('objective', '')}")
        lines.append(f"**Scoring:** {game.get('scoring', '')}")
        lines.append(f"**Victory:** {game.get('victory_condition', '')}")

    # Anti-boredom
    if anti_boredom:
        lines.append("\n---\n\n### 🔄 Anti-Boredom Strategies")
        for s in anti_boredom:
            lines.append(f"- {s}")

    # Thought Parking
    if thought_park and thought_park.get("enabled", True):
        lines.append("\n---\n\n### 🧠 Thought Parking Active")
        lines.append("If random ideas pop up, tell me and I'll park them for later.")
        lines.append("Categories: Tasks | Ideas | Worries | Random")

    # Rescue plan
    if rescue:
        lines.append(f"\n---\n\n### 🆘 If You Get Stuck\n{rescue}")

    return "\n".join(lines)
