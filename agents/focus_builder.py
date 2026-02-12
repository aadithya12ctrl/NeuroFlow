"""
NeuroFlow — Focus Environment Builder (Agent 6)
Generates personalized work environments based on task type, energy level,
and learned preferences.  Configures music, body doubling, ambient layers,
break activities, and thought parking.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from state import (
    NeuroFlowState,
    TaskEnvironment,
    BodyDoubleConfig,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Focus Environment Builder for NeuroFlow, an ADHD cognitive support system.

Your job: Given a task description and the user's current state, generate a PERSONALISED
work environment configuration that maximises focus and minimises friction.

## ADHD Environment Principles
1. **Music as Focus Anchor**: Match music BPM to task energy needs. Use the user's PREFERRED GENRE when specified.
2. **Body Doubling**: Virtual co-worker "Alex" increases accountability. Include timed check-ins.
3. **Ambient Layering**: ADHD brains often need sensory complexity (music + coffee shop + rain).
4. **Activity-Based Breaks**: NEVER say "take a break". Give SPECIFIC physical activities matched to task type.
5. **Thought Parking**: Enable intrusive thought capture so random ideas don't derail focus.

## BPM-MAPPED PLAYLIST — CRITICAL RULES
Generate a playlist of 4-6 SPECIFIC, REAL songs from the user's preferred genre.
Each song MUST be mapped to a work phase with STRICT BPM requirements:

| Phase | BPM Range | Purpose | When |
|-------|-----------|---------|------|
| 🚀 Startup | 130-150 BPM | HIGH energy to overcome initiation barrier — the hardest part | First 5 minutes |
| 🎯 Deep Focus | 70-90 BPM | LOW-MEDIUM, steady and calm for sustained concentration | Core work phase |
| 💪 Grind Phase | 140-170 BPM | HIGH energy to push through boring/hard middle section | When motivation dips |
| 🧘 Wind Down | 60-80 BPM | SLOW and reflective for reviewing and wrapping up | Last 5-10 minutes |

STRICT CONSTRAINTS:
- Startup BPM MUST be >= 130 (energising, hype)
- Deep Focus BPM MUST be <= 90 (calm, steady, not distracting)
- Grind Phase BPM MUST be >= 140 (re-energise, fight boredom)
- Wind Down BPM MUST be <= 80 (slow, reflective)
- Songs MUST be real, well-known songs from the user's specified genre
- If user doesn't specify a genre, use a mix of popular focus-friendly songs
- Map each song to the corresponding micro-step from the task breakdown

## Task-Type Rules
- **coding**: High-energy music (140+ BPM preferred), 25-min Pomodoro, body double ON, breaks = dance/jumping jacks
- **writing**: Calm music (80-100 BPM), 45-min deep work, breaks = read/doodle/walk
- **revision**: Upbeat music (120+ BPM), 15-min sprints (short = more wins), breaks = snack/TikTok/text friend
- **general**: Moderate BPM, 25-min Pomodoro, adjust based on energy

## Output Format (strictly JSON)
```json
{
  "music_style": "kpop|lo-fi|brown_noise|upbeat|silence|custom",
  "music_reasoning": "Why this music was chosen based on task + energy + user preference",
  "playlist": [
    {"section": "🚀 Startup", "song": "Song Title - Artist", "bpm": 140, "mapped_step": "What task step this plays during", "reason": "High BPM activation energy"},
    {"section": "🎯 Deep Focus (Part 1)", "song": "Song Title - Artist", "bpm": 80, "mapped_step": "What task step this plays during", "reason": "Steady calm for concentration"},
    {"section": "🎯 Deep Focus (Part 2)", "song": "Song Title - Artist", "bpm": 85, "mapped_step": "What task step this plays during", "reason": "Continues focus momentum"},
    {"section": "💪 Grind Phase", "song": "Song Title - Artist", "bpm": 150, "mapped_step": "What task step this plays during", "reason": "Energy boost for the hard middle"},
    {"section": "🧘 Wind Down", "song": "Song Title - Artist", "bpm": 70, "mapped_step": "What task step this plays during", "reason": "Calm transition to finish"}
  ],
  "timer_mode": "pomodoro|stopwatch|countdown",
  "timer_duration": 25,
  "ambient_layers": ["coffee_shop", "rain"],
  "body_double_enabled": true,
  "body_double_status": "Alex's current work status message",
  "break_activities": ["specific activity 1", "specific activity 2", "specific activity 3"],
  "thought_parking_enabled": true,
  "tools_enabled": ["notepad"]
}
```

IMPORTANT: When the user specifies a preferred genre (e.g., "K-Pop", "R&B", "EDM", "Bollywood", "Metal", "Classical"), generate the playlist USING ONLY REAL SONGS from that genre. The BPM values MUST follow the phase rules above — do NOT just decrease linearly. The pattern should be: HIGH → LOW → HIGH → LOW.

Return ONLY valid JSON. No markdown fences, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def focus_builder_node(state: dict) -> dict:
    """Generate a personalised focus environment configuration."""

    current_task = state.get("current_task", {})
    task_desc = current_task.get("description", "") if current_task else ""
    task_type = current_task.get("task_type", "general") if current_task else "general"
    cog = state.get("cognitive_state", {})
    energy = cog.get("energy_level", 7) if cog else 7
    focus = cog.get("focus_level", "medium") if cog else "medium"
    prefs = state.get("user_preferences", {})

    # Extract music genre preference from user input if present
    user_input = state.get("user_input", "")
    music_genre = "any"
    if "Preferred Music Genre:" in user_input:
        try:
            genre_part = user_input.split("Preferred Music Genre:")[1].split(",")[0].strip()
            if genre_part and genre_part.lower() != "any":
                music_genre = genre_part
        except Exception:
            pass

    # Build context prompt
    prompt = (
        f"## Task\n"
        f"Description: {task_desc}\n"
        f"Type: {task_type}\n\n"
        f"## User State\n"
        f"Energy: {energy}/10\n"
        f"Focus: {focus}\n\n"
        f"## User Preferences\n"
        f"Preferred music genre: {music_genre}\n"
        f"Body double preferred: {prefs.get('body_double_preferred', True)}\n"
        f"Body double name: {prefs.get('body_double_name', 'Alex')}\n\n"
        f"Generate the optimal focus environment for this task. "
        f"{'Use ' + music_genre + ' songs for the playlist! Find REAL songs in this genre with appropriate BPM for each section.' if music_genre != 'any' else 'Choose the best genre based on task type.'}"
    )

    env_config = {}
    focus_output = ""

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.6)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        env_config = json.loads(raw)

        # Build human-readable summary for the response generator
        music = env_config.get("music_style", "lo-fi")
        reasoning = env_config.get("music_reasoning", "")
        timer = env_config.get("timer_duration", 25)
        timer_mode = env_config.get("timer_mode", "pomodoro")
        breaks = env_config.get("break_activities", [])
        playlist = env_config.get("playlist", [])
        body_name = prefs.get("body_double_name", "Alex")
        body_enabled = env_config.get("body_double_enabled", True)
        body_status = env_config.get("body_double_status", "Getting ready to work...")
        ambient = env_config.get("ambient_layers", [])

        focus_parts = [
            f"🎵 **Music**: {music.replace('_', ' ').title()}" + (f" ({music_genre})" if music_genre != "any" else ""),
            f"   _Why_: {reasoning}" if reasoning else "",
            f"⏱️  **Timer**: {timer}-min {timer_mode.title()}",
        ]
        if ambient:
            focus_parts.append(f"🌊 **Ambient**: {', '.join(a.replace('_', ' ').title() for a in ambient)}")
        if body_enabled:
            focus_parts.append(f"👤 **Body Double**: {body_name} — \"{body_status}\"")
        
        # BPM-mapped playlist
        if playlist:
            focus_parts.append("\n🎶 **Your Focus Playlist** (BPM-mapped to each phase):")
            for track in playlist:
                section = track.get("section", "")
                song = track.get("song", "")
                bpm = track.get("bpm", "?")
                reason = track.get("reason", "")
                focus_parts.append(f"   • **{section}**: {song} ({bpm} BPM) — _{reason}_")
        
        if breaks:
            focus_parts.append("\n💃 **Break Activities**:")
            for b in breaks[:4]:
                focus_parts.append(f"   • {b}")

        focus_output = "\n".join(p for p in focus_parts if p)

    except Exception as e:
        print(f"[NeuroFlow] Focus builder error: {e}")
        traceback.print_exc()
        # Sensible defaults based on task type
        defaults = {
            "coding": {"music_style": "kpop", "timer_duration": 25,
                       "break_activities": ["💃 Dance to one song", "🏃 10 jumping jacks", "💧 Drink water"]},
            "writing": {"music_style": "lo-fi", "timer_duration": 45,
                        "break_activities": ["📖 Read one page", "✍️ Doodle 3 min", "🚶 Walk outside"]},
            "revision": {"music_style": "upbeat", "timer_duration": 15,
                         "break_activities": ["📱 Watch ONE short video", "🍿 Snack break", "💬 Text a friend"]},
        }
        d = defaults.get(task_type, {"music_style": "lo-fi", "timer_duration": 25,
                                      "break_activities": ["💧 Drink water", "🚶 Walk around"]})
        env_config = {
            "music_style": d["music_style"],
            "timer_mode": "pomodoro",
            "timer_duration": d["timer_duration"],
            "body_double_enabled": True,
            "body_double_status": "Getting ready to work with you!",
            "break_activities": d["break_activities"],
            "ambient_layers": [],
            "thought_parking_enabled": True,
            "tools_enabled": ["notepad"],
        }
        focus_output = f"🎵 {d['music_style'].title()} music | ⏱️ {d['timer_duration']}-min Pomodoro | 👤 Alex is ready"

    # Update the current task's environment
    updated_task = dict(current_task) if current_task else {}
    env = updated_task.get("environment", {})
    env.update({
        "music_style": env_config.get("music_style", "lo-fi"),
        "music_reasoning": env_config.get("music_reasoning", ""),
        "timer_mode": env_config.get("timer_mode", "pomodoro"),
        "timer_duration": env_config.get("timer_duration", 25),
        "body_double": {
            "enabled": env_config.get("body_double_enabled", True),
            "name": prefs.get("body_double_name", "Alex"),
            "status": env_config.get("body_double_status", "Getting ready..."),
            "check_in_intervals": [15, 30, 45],
            "presence_indicator": "🟢",
        },
        "ambient_layers": env_config.get("ambient_layers", []),
        "break_activities": env_config.get("break_activities", []),
        "playlist": env_config.get("playlist", []),
        "thought_parking_enabled": env_config.get("thought_parking_enabled", True),
        "tools_enabled": env_config.get("tools_enabled", ["notepad"]),
    })
    updated_task["environment"] = env

    return {
        "current_task": updated_task,
        "focus_output": focus_output,
    }
