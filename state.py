"""
NeuroFlow State Schema (v3)
Defines the LangGraph state used across all agents.
Supports 7-agent architecture with Dopamine Economy + Focus Environment Builder.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Pydantic helper models (used inside the state)
# ---------------------------------------------------------------------------

class CrashPrediction(BaseModel):
    likelihood: float = 0.0          # 0-1
    estimated_minutes: int = 999     # minutes until predicted crash


class CognitiveState(BaseModel):
    focus_level: Literal["low", "medium", "high", "hyperfocus"] = "medium"
    energy_level: int = 7            # 0-10
    dopamine_balance: int = 50       # 0-100
    crash_prediction: CrashPrediction = Field(default_factory=CrashPrediction)


class InteractionMetrics(BaseModel):
    typing_speed_baseline: float = 0.0
    current_typing_speed: float = 0.0
    avg_message_length: int = 0
    response_time_trend: Literal["increasing", "stable", "decreasing"] = "stable"
    last_break: Optional[str] = None          # ISO timestamp
    message_lengths: list[int] = Field(default_factory=list)
    response_times: list[float] = Field(default_factory=list)


class PatternDetection(BaseModel):
    current_pattern: Literal[
        "none", "avoidance", "distraction", "paralysis",
        "productive", "perfectionism"
    ] = "none"
    pattern_start_time: Optional[str] = None  # ISO timestamp
    interventions_attempted: list[str] = Field(default_factory=list)
    sentiment_trajectory: list[float] = Field(default_factory=list)  # -1 to 1


# ---------------------------------------------------------------------------
# Dopamine Economy Models
# ---------------------------------------------------------------------------

class DopamineTransaction(BaseModel):
    """Single dopamine budget transaction."""
    event_type: str = ""              # e.g. task_started, doom_scrolled
    points: int = 0                   # positive = gain, negative = cost
    timestamp: str = ""               # ISO timestamp
    description: str = ""


class DopamineEconomy(BaseModel):
    """Daily dopamine budget tracker (0-100)."""
    daily_balance: int = 100
    transactions: list[DopamineTransaction] = Field(default_factory=list)
    forecast: str = ""                # "☀️ High energy" / "⛈️ Low energy"
    next_reward_minutes: int = 0      # minutes until next variable reward
    reward_schedule: list[int] = Field(default_factory=list)  # e.g. [8,15,27,35]


# ---------------------------------------------------------------------------
# Focus Environment Models
# ---------------------------------------------------------------------------

class BodyDoubleConfig(BaseModel):
    """Virtual co-worker (body doubling) configuration."""
    enabled: bool = True
    name: str = "Alex"
    status: str = "Getting ready to work..."
    check_in_intervals: list[int] = Field(default_factory=lambda: [15, 30, 45])
    presence_indicator: str = "🟢"


class TaskEnvironment(BaseModel):
    music_style: str = "lo-fi"        # lo-fi, kpop, brown_noise, silence, upbeat
    music_reasoning: str = ""         # why this music was chosen
    timer_mode: str = "pomodoro"      # pomodoro, stopwatch, countdown
    timer_duration: int = 25          # minutes
    tools_enabled: list[str] = Field(default_factory=lambda: ["notepad"])
    video_url: str = ""               # optional YouTube link
    layout: str = "focused"           # focused, split
    body_double: BodyDoubleConfig = Field(default_factory=BodyDoubleConfig)
    ambient_layers: list[str] = Field(default_factory=list)   # ["coffee_shop", "rain"]
    break_activities: list[str] = Field(default_factory=list)  # specific per task type
    thought_parking_enabled: bool = True


# ---------------------------------------------------------------------------
# Thought Parking
# ---------------------------------------------------------------------------

class ParkedThought(BaseModel):
    """An intrusive thought captured mid-task."""
    thought: str = ""
    category: Literal["task", "idea", "worry", "random"] = "random"
    captured_at: str = ""             # ISO timestamp
    resurface_at: str = "next_break"  # when to show again


# ---------------------------------------------------------------------------
# Task Info
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: str
    description: str
    task_type: str = "general"        # coding, writing, revision, general
    start_time: str | None = None
    estimated_duration: int = 30
    realistic_duration: int = 0       # after ADHD multiplier applied
    context_package: dict = {}        # The full generated advice
    environment: TaskEnvironment = Field(default_factory=TaskEnvironment)
    progress_milestones: list[str] = Field(default_factory=list)
    completed_milestones: list[str] = Field(default_factory=list)
    progress_percent: int = 0
    thought_parking_lot: list[ParkedThought] = Field(default_factory=list)
    dopamine_checkpoints: list[dict] = Field(default_factory=list)
    initiation_ritual: list[str] = Field(default_factory=list)
    anti_repetition_mode: str = ""    # bug_hunter, speed_run, etc.


class UserPreferences(BaseModel):
    work_style: str = "balanced"
    preferred_break_duration: int = 5         # minutes
    notification_sensitivity: Literal["low", "medium", "high"] = "medium"
    preferred_music: dict = Field(default_factory=lambda: {
        "coding": "kpop",
        "writing": "lo-fi",
        "revision": "upbeat"
    })
    body_double_preferred: bool = True
    body_double_name: str = "Alex"


# ---------------------------------------------------------------------------
# Root LangGraph State  (TypedDict so LangGraph can merge updates)
# ---------------------------------------------------------------------------

class NeuroFlowState(TypedDict, total=False):
    # Session
    session_id: str
    session_start: str          # ISO timestamp
    interaction_count: int

    # Messages (LangGraph message list with automatic merging)
    messages: Annotated[list, add_messages]

    # Current user input
    user_input: str

    # Routing
    intent: str                 # start_task | stuck | distracted | check_in | general_chat
    priority: bool              # urgent intervention needed?

    # Current task
    current_task: dict          # serialised TaskInfo
    task_queue: list[dict]

    # Cognitive state
    cognitive_state: dict       # serialised CognitiveState

    # Interaction metrics
    interaction_metrics: dict   # serialised InteractionMetrics

    # Pattern detection
    pattern_detection: dict     # serialised PatternDetection

    # Dopamine economy
    dopamine_economy: dict      # serialised DopamineEconomy

    # User preferences
    user_preferences: dict      # serialised UserPreferences

    # Agent outputs (collected before response generation)
    cognitive_output: str
    context_output: str
    pattern_output: str
    time_output: str
    focus_output: str           # Focus Environment Builder output
    dopamine_output: str        # Dopamine Economy Manager output

    # Advanced graph control fields
    pattern_escalation_level: int   # 0=initial, 1=escalated, 2=max (cyclic loop counter)
    response_retry_count: int       # Self-correction loop counter (max 1 retry)
    quality_score: float            # 0.0-1.0 quality assessment from quality gate
    needs_human_approval: bool      # Human-in-the-loop flag for task plans

    # Final response to user
    response: str
