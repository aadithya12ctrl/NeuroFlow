# Product Requirements Document: NeuroFlow
## ADHD Cognitive Support System

---

## 🎯 Executive Summary

**Project Name:** NeuroFlow  
**Timeline:** 1 week  
**Purpose:** Portfolio project showcasing LangGraph multi-agent architecture + genuinely useful ADHD support tool  
**Target Audience:** Potential employers (primary), startup investors (secondary)  
**Tech Stack:** LangGraph + Gemini Flash 1.5 + Streamlit + ChromaDB + SQLite  
**Scope:** 5-6 deeply implemented agents, locally hosted web app  
**Design Theme:** Warm autumn aesthetic with muted olives, oranges, browns, and creams

---

## 🧠 Problem Statement

ADHD individuals face four critical challenges:

1. **Initiation Paralysis** - Inability to start tasks despite wanting to
2. **Time Blindness** - Poor time estimation and awareness
3. **Distraction Susceptibility** - Constant context switching and interruptions
4. **Repetition Aversion** - Extreme difficulty with recurring/revision tasks

Current task management tools are built for neurotypical users and fail to address these neurological differences.

---

## 💡 Solution Overview

NeuroFlow is a predictive, adaptive multi-agent system that:

- Predicts cognitive crashes before they happen
- Designs custom work environments for each task
- Breaks negative behavioral loops in real-time
- Reduces friction for task initiation and context recovery
- Makes repetitive tasks psychologically bearable

**Core Innovation:** Instead of reactive task management, NeuroFlow uses behavioral pattern analysis to intervene before problems occur.

---

## 🏗️ System Architecture

### Multi-Agent Framework (5 Core Agents)

```
User Input → Session Manager → State Graph
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
Cognitive       Context         Pattern
Monitor         Builder         Detective
    ↓               ↓               ↓
Predicts        Designs         Breaks
crashes         environ.        loops
    ↓               ↓               ↓
    └───────────────┼───────────────┘
                    ↓
          Response Generator
                    ↓
          Update State & UI
```

---

## 🤖 Agent Specifications

### Agent 1: Session Manager (Router/Orchestrator)

**Role:** Entry point, routes to appropriate agents, maintains session state

**Responsibilities:**
- Analyzes user input and determines intent
- Routes to specialized agents based on context
- Manages conversation flow
- Maintains global state consistency

**LangGraph Implementation:**
- Acts as the main router node
- Conditional edges based on user intent classification
- Simple intent categories: `start_task`, `stuck`, `distracted`, `check_in`, `general_chat`

**Inputs:**
- User message
- Current session state
- Time of day
- Active task (if any)

**Outputs:**
- Routing decision
- Updated session context
- Priority flag (urgent intervention needed?)

---

### Agent 2: Cognitive State Predictor

**Role:** Monitors interaction patterns to predict focus crashes and cognitive overload

**Responsibilities:**
- Tracks typing speed, message length, response time patterns
- Detects early warning signs of cognitive decline
- Predicts optimal break timing
- Identifies hyperfocus states (both productive and unproductive)

**Intelligence Level:** 70% rules, 30% pattern detection

**Pattern Detection Rules:**

```python
# Focus crash indicators
- Typing speed decreasing >30% from baseline
- Message length getting shorter progressively
- Response time increasing
- Typo frequency increasing
- Time since last break >90 minutes

# Hyperfocus indicators
- Rapid continuous interaction
- No breaks for extended period
- Task switching rate = 0
- Message coherence high but single-topic
```

**Database Schema:**

```sql
interaction_metrics:
  - timestamp
  - typing_speed (chars/sec)
  - message_length
  - response_time_seconds
  - typo_count
  - session_duration
  - current_task_id
```

**LangGraph Implementation:**
- Runs as background node on every user interaction
- Updates interaction metrics in state
- Conditional edge: if crash predicted → trigger intervention
- Stores historical patterns in ChromaDB for similarity search

**Intervention Types:**

| Prediction | Intervention |
|------------|-------------|
| Predicted crash in 15-20 min | "I'm noticing your energy dipping. Want to take a strategic break now?" |
| Hyperfocus on right task | Protect it, minimal interruptions |
| Hyperfocus on wrong task | Wait for natural pause, then redirect |
| Cognitive overload detected | Suggest task simplification |

---

### Agent 3: Context Architect

**Role:** Designs complete work environments tailored to specific tasks and cognitive states

**Responsibilities:**
- Analyzes task requirements (focus level, creativity needed, duration)
- Matches user's current energy level to task demands
- Creates "focus packages" with multiple elements
- Generates task initiation sequences to overcome paralysis

**Context Package Components:**

1. **Task Breakdown** - Micro-steps with time estimates
2. **Environment Setup** - Workspace preparation checklist
3. **Focus Timer** - Adaptive Pomodoro based on task type
4. **Initiation Ritual** - Specific sequence to begin (critical for ADHD)
5. **Progress Milestones** - Dopamine checkpoints

**Task Analysis Dimensions:**

```python
{
  "cognitive_load": "low | medium | high",
  "creativity_required": "analytical | balanced | creative",
  "estimated_duration": minutes,
  "interruptibility": "cannot_interrupt | flexible | async",
  "repetition_factor": 0-10  # Higher = more repetitive/boring
}
```

**Anti-Repetition Strategies:**
- Gamification elements for revision tasks
- Varied presentation formats
- Micro-rewards at unusual intervals (variable ratio reinforcement)
- "Disguise" repetitive tasks as different activities
- Break into smallest possible chunks with different contexts

**LangGraph Implementation:**
- Triggered when user wants to start a task
- Subgraph for task analysis → environment design → package assembly
- Uses ChromaDB to retrieve similar past successful contexts
- Generates personalized initiation ritual

**Example Context Package:**

**Task:** Review Python documentation (HIGH repetition aversion)

```
🎯 MISSION: Extract 5 interesting code patterns

⏱️ Format: 5x 10-minute discovery sprints

🎵 Vibe: Lo-fi hip hop beats

📋 Initiation Ritual:
  1. Close all tabs except docs
  2. Get water
  3. Put phone in drawer
  4. Set timer for 10 min
  5. START by searching for "decorators" (specific first step)

🎮 Game Mode: "Pattern Hunter"
  - Find 1 pattern per sprint
  - Each pattern = 1 point
  - 5 points = victory condition
  - Make it interesting: look for WEIRD patterns

✅ Micro-milestones:
  - 10 min: First pattern found (+dopamine)
  - 20 min: Second pattern (+dopamine)
  - 30 min: Halfway celebration
  - 50 min: Final pattern (+victory)
```

---

### Agent 4: Pattern Interrupt Specialist

**Role:** Detects negative behavioral loops and deploys strategic interrupts

**Responsibilities:**
- Identifies avoidance patterns (task switching, distraction seeking)
- Detects doom scrolling / procrastination spirals
- Recognizes productive vs unproductive task switching
- Deploys context-appropriate interventions

**Detectable Patterns:**

#### 1. Avoidance Spiral
**Indicators:**
- User keeps asking about different tasks without starting any
- Multiple task breakdowns requested but none initiated
- Questions getting progressively less specific
- Time passing without action

**Intervention:** "I notice we've planned 3 tasks but haven't started any. Let's use Chaos Mode: pick ONE random micro-task and do it for just 5 minutes. No thinking, just start."

#### 2. Productive Procrastination
**Indicators:**
- Doing legitimate tasks but avoiding the important one
- Over-planning, over-researching, over-preparing
- "Productive" activities that are actually avoidance

**Intervention:** "You've been preparing for 25 minutes. That's perfect preparation. Time to act. What's the absolute smallest first step?"

#### 3. Distraction Cascade
**Indicators:**
- Rapid topic switching in conversation
- User mentions getting sidetracked
- Time gaps between messages increasing
- Off-topic tangents

**Intervention:** "Welcome back! You were working on [TASK]. Here's exactly where you were: [CONTEXT]. Continue or switch?"

#### 4. Decision Paralysis
**Indicators:**
- Asking for comparison between options repeatedly
- "Which should I..." questions
- Analysis paralysis language
- No decision after 10+ minutes

**Intervention:** "I'm going to make this easy. Do [OPTION A] first. Not because it's better, but because deciding wastes more energy than doing. Start it for 10 minutes. If it sucks, switch."

**Pattern Interrupt Techniques:**
- **Chaos Mode:** Random task assignment to break paralysis
- **Anchor Return:** Remind of original intent + exact context
- **Decision Elimination:** Remove choices temporarily
- **Absurdist Interrupt:** Unexpected question to snap attention
- **Physical Interrupt:** Suggest movement/break to reset

**LangGraph Implementation:**
- Continuously analyzes conversation history (last 10 interactions)
- Pattern matching against known anti-patterns
- Conditional routing: pattern detected → select interrupt strategy
- Uses ChromaDB to find similar past patterns and successful interventions

**Database Schema:**

```sql
pattern_events:
  - timestamp
  - pattern_type (avoidance, distraction, paralysis)
  - context (what task, what state)
  - intervention_used
  - success (boolean - did it work?)
  - user_response
```

---

### Agent 5: Time Reality Agent

**Role:** Combats time blindness through realistic estimation and continuous awareness

**Responsibilities:**
- Provides realistic time estimates based on user's historical data
- Maintains time awareness during tasks
- Predicts task duration based on similar past tasks
- Creates calibrated schedules that account for ADHD time perception

**Core Features:**

#### 1. Reality-Based Estimation
```python
# User says: "This will take 30 minutes"
# Agent calculates:
historical_similar_tasks = query_chromadb(task_description)
actual_average = mean([task.actual_duration for task in historical_similar_tasks])
adhd_multiplier = 1.5  # Research-based average

realistic_estimate = user_estimate * adhd_multiplier
# Returns: "Based on similar tasks, this usually takes you 45-60 minutes"
```

#### 2. Time Anchors
- Periodic check-ins: "You've been working for 20 minutes (you estimated 15)"
- Remaining time updates: "15 minutes left until your deadline"
- Hyperfocus warnings: "You've been in flow for 2 hours - time for a break?"

#### 3. Schedule Builder
Creates realistic daily plans:
- Accounts for transition time between tasks (15-20 min)
- Includes mandatory breaks
- Leaves buffer time for unexpected
- Adjusts based on energy levels throughout day

**Database Schema:**

```sql
task_history:
  - task_id
  - description
  - estimated_duration
  - actual_duration
  - completion_date
  - energy_level_at_start
  - interruptions_count
  
time_blocks:
  - block_id
  - start_time
  - end_time
  - task_id
  - actual_productivity (1-10)
```

**LangGraph Implementation:**
- Background timer tracking active tasks
- Periodic state updates with elapsed time
- Conditional alerts based on time thresholds
- Historical analysis subgraph for estimation

**Time Awareness Interventions:**

| Scenario | Intervention |
|----------|-------------|
| User estimate too optimistic | "Similar tasks took you X minutes. Want to plan for Y instead?" |
| Running over estimate | "You're at 35 min (estimated 30). Need 15 more or wrap up?" |
| Hyperfocus detected | "You've been focused for 90 min. Quick 5-min break?" |
| Deadline approaching | "30 minutes until your meeting. Time to wrap up?" |

---

## 📊 State Management

### Global State Schema

```python
{
  "session": {
    "session_id": "uuid",
    "start_time": "timestamp",
    "user_id": "string",
    "interaction_count": "int"
  },
  
  "current_task": {
    "task_id": "uuid",
    "description": "string",
    "start_time": "timestamp",
    "estimated_duration": "int (minutes)",
    "context_package": "object",
    "progress_milestones": ["list"],
    "completed_milestones": ["list"]
  },
  
  "cognitive_state": {
    "focus_level": "low | medium | high | hyperfocus",
    "energy_level": "int (0-10)",
    "dopamine_balance": "int (0-100)",
    "crash_prediction": {
      "likelihood": "float (0-1)",
      "estimated_minutes": "int"
    }
  },
  
  "interaction_metrics": {
    "typing_speed_baseline": "float",
    "current_typing_speed": "float",
    "avg_message_length": "int",
    "response_time_trend": "increasing | stable | decreasing",
    "last_break": "timestamp"
  },
  
  "pattern_detection": {
    "current_pattern": "none | avoidance | distraction | paralysis | productive",
    "pattern_start_time": "timestamp",
    "interventions_attempted": ["list"]
  },
  
  "task_queue": ["list of task objects"],
  
  "user_preferences": {
    "work_style": "string",
    "preferred_break_duration": "int",
    "notification_sensitivity": "low | medium | high"
  }
}
```

### State Update Rules

1. **Every user interaction:** Update interaction_metrics
2. **Every 5 minutes:** Run cognitive state prediction
3. **Task start:** Create context_package, initialize milestones
4. **Task complete:** Archive to history, update patterns
5. **Pattern detected:** Log pattern_event, trigger intervention

---

## 🎨 User Interface Design

### Design System

**Color Palette:**
- **Cream:** `#F5F0E8` (backgrounds)
- **Taupe:** `#C9B8A8` (borders, subtle elements)
- **Olive:** `#8FA67E` (primary actions, progress)
- **Burnt Orange:** `#C8763F` (secondary actions, alerts)
- **Rust:** `#A84C32` (warnings, urgent items)
- **Terracotta:** `#B85C4F` (celebrations, achievements)
- **Dark Brown:** `#3E3028` (primary text)
- **Medium Brown:** `#6B5B52` (secondary text)

**Typography:**
- **Headings:** Inter, 24-32px, bold
- **Body:** Inter, 16px, regular
- **UI Elements:** Inter, 14px, medium
- **Monospace (timers):** JetBrains Mono, 18px

---

### Layout Structure

```
┌────────────────────────────────────────────────────────┐
│                    NeuroFlow Logo                       │
├────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐ │
│  │          🧠 State Dashboard (4 columns)          │ │
│  │  Focus State │ Energy │ Dopamine │ Session Time  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐ │
│  │              📋 Active Task Card                  │ │
│  │  ┌────────────────────────────────────────────┐  │ │
│  │  │ Task: Review Python Documentation          │  │ │
│  │  │ Progress: ████████░░░░░░░░░░ 45%          │  │ │
│  │  │ Next: Find third code pattern              │  │ │
│  │  │ Started: 2:30 PM • 25 min • 15 min left   │  │ │
│  │  │ [Continue] [Pause] [Need Help]             │  │ │
│  │  └────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐ │
│  │              💬 NeuroFlow Assistant              │ │
│  │  ┌────────────────────────────────────────────┐  │ │
│  │  │                                            │  │ │
│  │  │  🧠 NeuroFlow:                             │  │ │
│  │  │  I notice you've been working for 45       │  │ │
│  │  │  minutes. Your typing speed is starting    │  │ │
│  │  │  to slow down. Want to take a strategic    │  │ │
│  │  │  5-minute break now?                       │  │ │
│  │  │  2 min ago                                 │  │ │
│  │  │                                            │  │ │
│  │  │                          You:              │  │ │
│  │  │          Not yet, almost done with this    │  │ │
│  │  │                             section        │  │ │
│  │  │                            Just now        │  │ │
│  │  │                                            │  │ │
│  │  │  🧠 NeuroFlow:                             │  │ │
│  │  │  Got it! I'll check in again in 15         │  │ │
│  │  │  minutes. 💪 You're doing great!           │  │ │
│  │  │  Just now                                  │  │ │
│  │  │                                            │  │ │
│  │  └────────────────────────────────────────────┘  │ │
│  │  ┌────────────────────────────────────────────┐  │ │
│  │  │ Type your message... [Send]                │  │ │
│  │  └────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐ │
│  │              ⚡ Quick Actions                     │ │
│  │  [🎯 Start New Task]  [🆘 I'm Stuck]             │ │
│  │  [☕ Take Break]       [📊 Review Progress]       │ │
│  └──────────────────────────────────────────────────┘ │
│                                                          │
└────────────────────────────────────────────────────────┘
```

---

### UI Component Specifications

#### 1. State Dashboard (Top Card)

**Purpose:** At-a-glance cognitive and motivational status

**Visual Design:**
- Cream background card with taupe border
- Rounded corners (12px border-radius)
- Subtle shadow for depth
- Four equal-width columns

**Focus State Indicator:**
- Large emoji/icon visual (🟢🟡🔴🟣)
- Color-coded background pill
- Text label below
- Smooth transitions between states

**Energy Level:**
- Horizontal progress bar
- Burnt orange fill color
- Lightning bolt emoji (⚡)
- Number display (7/10)

**Dopamine Balance:**
- Horizontal progress bar
- Gradient from rust (low) to terracotta (high)
- Flame emoji (🔥)
- Number display (65/100)

**Session Time:**
- Auto-updating counter
- Clock emoji (⏱️)
- Format: "45min" or "1h 23min"

---

#### 2. Active Task Card

**Purpose:** Detailed view of current work with context and actions

**Visual Design:**
- Warm white background
- Slightly larger than dashboard card
- Clear visual hierarchy with sections

**Elements:**

**Task Title:** Large, bold, dark brown text

**Progress Bar:**
- Olive green fill for completed portion
- Light cream/taupe for remaining
- Percentage label
- Block/segment style (not smooth) for discrete steps

**Next Step:**
- Highlighted in a light olive background pill
- Arrow emoji (➡️)
- Slightly larger text

**Metadata Row:**
- Started time • Estimated duration • Time remaining
- Small text, medium brown color
- Separated by bullet points

**Action Buttons:**
- Primary: Deep olive with white text
- Secondary: Burnt orange with white text
- Help: Rust color with white text
- Icons + text labels
- Rounded buttons with hover effects

---

#### 3. Chat Interface

**Purpose:** Natural conversation with the AI assistant

**Visual Design:**
- Largest card on page
- Scrollable message area
- Fixed input at bottom

**Message Bubbles:**

**AI Messages:**
- Light olive background (#8FA67E at 20% opacity)
- Dark brown text
- Left-aligned
- Robot/brain emoji (🧠) as avatar
- Timestamp in small text below

**User Messages:**
- Burnt orange background (#C8763F at 20% opacity)
- Dark brown text
- Right-aligned
- No avatar (cleaner)
- Timestamp in small text below

**Special Message Types:**
- **Urgent Interventions:** Red/rust left border stripe
- **Celebrations:** Terracotta background with confetti emoji
- **Context Packages:** Structured format with sections, icons, checkboxes

**Input Area:**
- Warm white background
- Taupe border
- Large, comfortable text area
- Send button in deep olive
- Placeholder text in medium brown

---

#### 4. Quick Actions Bar

**Purpose:** One-click access to common actions

**Visual Design:**
- Horizontal button row
- Equal-width buttons
- Icons + text labels
- Spaced evenly

**Button Styles:**
- Deep olive background
- White text
- Rounded corners (8px)
- Hover: Lighten to medium olive
- Active: Slightly darker, scale down (0.98)
- Shadow on hover for depth

**Action Types:**
- **Start New Task (🎯)** - Primary action
- **I'm Stuck (🆘)** - Help action
- **Take Break (☕)** - Wellness action
- **Review Progress (📊)** - Reflection action

---

### Interaction Design

**Micro-interactions:**
- Smooth transitions (200ms ease-in-out)
- Button hover states with slight lift
- Progress bar animations
- Gentle fade-ins for new messages
- Pulse animation for urgent alerts
- Confetti or celebration animation for achievements

**Loading States:**
- Olive-colored spinner
- "Thinking..." text in medium brown
- Pulsing opacity on AI avatar during generation

**Empty States:**
- Warm illustrations in autumn colors
- Friendly copy
- Clear call-to-action

**Error States:**
- Rust-colored border on affected element
- Icon + clear error message
- Suggested action to resolve

---

### Accessibility Considerations

**For ADHD Users:**
- Minimal clutter: Only essential information visible
- Clear visual hierarchy: Size, color, spacing guide attention
- Consistent patterns: Same UI elements work the same way
- Collapsible sections: Reduce overwhelm, expand on demand
- Keyboard shortcuts: Reduce mouse dependency
- Focus indicators: Clear visual feedback for keyboard navigation

**General Accessibility:**
- Color contrast: Minimum 4.5:1 ratio (WCAG AA)
- Text sizing: Scalable with browser zoom
- Screen reader support: Proper semantic HTML and ARIA labels
- Reduced motion option: Respect prefers-reduced-motion
- Clear language: Simple, direct copy

---

### Responsive Behavior

**Desktop (Primary):**
- Two-column layout possible
- All features visible
- Comfortable spacing

**Tablet (If time permits):**
- Single column, full-width cards
- Maintained card hierarchy
- Touch-friendly button sizes

**Mobile (Low priority for v1):**
- Simplified view
- Collapsible cards
- Bottom navigation for quick actions

---

### Visual Enhancements

**Subtle Texture:**
- Very light paper texture on cream background
- Adds warmth without distraction
- Barely perceptible (5-10% opacity)

**Shadows:**
- Soft, warm shadows (not pure black)
- Shadow color: `rgba(62, 48, 40, 0.08)`
- Small elevation for cards: `0 2px 8px`
- Medium elevation for modals: `0 4px 16px`

**Icons:**
- Lucide icons or Phosphor icons (consistent style)
- Monochrome, matching text color
- 20-24px size for UI elements
- Larger (32-48px) for empty states

**Illustrations (Optional):**
- Simple line art in olive tones
- Used for empty states
- Celebrate achievements
- Minimal, not distracting

---

### Animation Principles

**Purpose-Driven:**
- Draw attention to important changes
- Provide feedback for actions
- Create delight for achievements
- Never purely decorative

**ADHD-Friendly:**
- Short duration (200-300ms)
- Not looping indefinitely (except loading)
- Can be disabled
- Not competing for attention

**Examples:**
- Task completion: Progress bar fills + checkmark appears
- Focus state change: Color transition + subtle scale
- New message: Gentle fade-in from top
- Dopamine gain: Number count-up + sparkle

---

### Dark Mode (Bonus Feature)

**Autumn Dark Palette:**
- Background: Deep charcoal brown (`#2A2420`)
- Cards: Slightly lighter brown (`#3E3028`)
- Text: Cream (`#F5F0E8`)
- Accents: Same olive/orange/rust but 20% more saturated
- Maintains autumn warmth, not cool grays

---

## 🗄️ Data Storage

### Database Architecture

**SQLite (Structured Data):**
- User profiles
- Task history
- Interaction metrics
- Pattern events
- Time tracking data

**ChromaDB (Vector Storage):**
- Conversation history embeddings
- Task context embeddings
- Successful intervention patterns
- Similar task retrieval

### Key Collections

#### tasks_collection
```python
{
  "task_id": "uuid",
  "description": "string",
  "embedding": [vector],
  "metadata": {
    "created_at": "timestamp",
    "cognitive_load": "string",
    "repetition_factor": "int",
    "actual_duration": "int",
    "success_rating": "int"
  }
}
```

#### interventions_collection
```python
{
  "intervention_id": "uuid",
  "pattern_type": "string",
  "intervention_text": "string",
  "embedding": [vector],
  "metadata": {
    "success": "boolean",
    "context": "string",
    "timestamp": "timestamp"
  }
}
```

---

## 🚀 Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)
- LangGraph state management setup
- SQLite + ChromaDB integration
- Session Manager agent (router)
- Basic Streamlit UI shell

### Phase 2: Cognitive Monitoring (Days 2-3)
- Cognitive State Predictor agent
- Interaction metrics tracking
- Pattern detection rules
- Database schemas

### Phase 3: Context & Intervention (Days 3-5)
- Context Architect agent
- Pattern Interrupt Specialist agent
- Context package generation
- Intervention strategies

### Phase 4: Time Management (Day 5)
- Time Reality Agent
- Historical analysis
- Realistic estimation
- Time awareness features

### Phase 5: UI Polish (Days 5-7)
- Full Streamlit interface
- Autumn design system
- Animations and micro-interactions
- Accessibility features

### Phase 6: Testing & Demo (Day 7)
- End-to-end testing
- Demo scenario creation
- Documentation
- Portfolio presentation prep

---

## 📈 Success Metrics

### Technical Metrics
- Agent response time < 500ms
- State persistence working 100%
- No critical bugs in demo scenarios
- Clean, documented code

### Portfolio Metrics
- Demonstrates LangGraph expertise
- Shows multi-agent orchestration
- Production-quality UI
- Solves real problem

### User Experience Metrics (If Testing)
- Successful task initiations
- Break compliance rate
- Pattern interrupt acceptance
- Time estimate accuracy improvement

---

## 🎯 Out of Scope (v1)

- Mobile apps (native iOS/Android)
- Calendar integration
- Email/Slack notifications
- Team/collaboration features
- Cloud hosting
- User authentication
- Payment processing
- Machine learning models (beyond pattern matching)
- Voice interface
- Browser extension

---

## 📝 Technical Notes

### LangGraph Patterns
- Use StateGraph for main orchestration
- Conditional edges for routing
- Parallel node execution where possible
- Checkpointing for state persistence

### Performance Considerations
- Cache ChromaDB embeddings
- Batch database writes
- Lazy load conversation history
- Optimize Streamlit rerenders

### Error Handling
- Graceful degradation for agent failures
- Fallback responses for timeout
- State recovery mechanisms
- User-friendly error messages

---

## 📚 References

### ADHD Research
- Time blindness studies
- Executive function research
- Dopamine reward systems
- Task initiation barriers

### Technical Documentation
- LangGraph docs
- Gemini API docs
- Streamlit best practices
- ChromaDB usage patterns

---

## ✅ Definition of Done

A working demo where:
1. User can start a task and receive a context package
2. System predicts and intervenes for cognitive decline
3. Pattern detection identifies and breaks avoidance loops
4. Time tracking provides realistic estimates
5. UI is polished with autumn aesthetic
6. Code is clean, documented, and portfolio-ready
7. Demo script showcases all 5 agents
8. README explains architecture and showcases results

---

**Document Version:** 1.0  
**Last Updated:** 2025  
**Author:** Product Team  
**Status:** Ready for Implementation
