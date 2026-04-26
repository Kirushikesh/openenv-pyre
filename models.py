"""
Data models for the Pyre Environment.

Pyre is a first-person crisis navigation environment: an LLM agent navigates
a burning building, manages fire spread through door control, and survives
under partial observability with a real health system.

Cell encoding (cell_grid):
  0 = floor
  1 = wall
  2 = door_open
  3 = door_closed
  4 = exit
  5 = obstacle (burned-out or structural)
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Map state (numerical grid snapshot — for UI / visualization consumers)
# ---------------------------------------------------------------------------

class PyreMapState(BaseModel):
    """Full numerical state of the environment at a single timestep.

    Intended for UI rendering and external tooling. Not filtered by visibility —
    the complete ground-truth grid is included so a renderer can draw the full
    map with fog-of-war applied client-side using `visible_cells`.

    Grid layout: flat row-major lists of length grid_w * grid_h.
    Index formula: i = y * grid_w + x
    """

    # Dimensions & identity
    grid_w: int = Field(..., description="Grid width in cells")
    grid_h: int = Field(..., description="Grid height in cells")
    template_name: str = Field(..., description="Floor plan template in use")
    episode_id: str = Field(..., description="Unique ID for this episode")
    step_count: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps allowed this episode")

    # Full grids (flat, row-major)
    cell_grid: List[int] = Field(
        ...,
        description="Cell types: 0=floor 1=wall 2=door_open 3=door_closed 4=exit 5=obstacle",
    )
    fire_grid: List[float] = Field(
        ..., description="Fire intensity per cell, 0.0 (none) → 1.0 (fully burning)"
    )
    smoke_grid: List[float] = Field(
        ..., description="Smoke intensity per cell, 0.0 (clear) → 1.0 (dense)"
    )

    # Agent
    agent_x: int = Field(..., description="Agent column (0-indexed from west)")
    agent_y: int = Field(..., description="Agent row (0-indexed from north)")
    agent_alive: bool = Field(..., description="Whether the agent is alive")
    agent_evacuated: bool = Field(..., description="Whether the agent has escaped")
    agent_health: float = Field(..., description="Agent health 0–100")

    # Visibility (fog-of-war)
    visible_cells: List[List[int]] = Field(
        ..., description="[[x, y], ...] cells visible to the agent this step"
    )

    # Points of interest (redundant with cell_grid but explicit for convenience)
    exit_positions: List[List[int]] = Field(
        ..., description="[[x, y], ...] coordinates of all exit cells"
    )
    door_registry: Dict[str, List[int]] = Field(
        ..., description="door_id → [x, y] position for every door"
    )

    # Episode fire parameters
    fire_spread_rate: float = Field(..., description="Probability of fire spreading per step")
    wind_dir: str = Field(..., description="Wind direction affecting fire spread")
    humidity: float = Field(..., description="Humidity level (higher = slower spread)")

    # Visual decorations: "x,y" → item type ("desk"|"chair"|"filing"|"plant"|"table")
    # Fixed per template, purely cosmetic — no gameplay or physics effect.
    furniture_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Cosmetic furniture: 'x,y' → item type for UI rendering",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PyreAction(Action):
    """Agent action for the Pyre environment.

    action:     "move" | "door" | "wait" | "look"
    direction:  "north"|"south"|"east"|"west"  — used by move and look
    target_id:  door ID ("door_3")             — used by door
    door_state: "open" | "close"               — used by door

    look: scan up to 5 cells in one direction; returns per-cell descriptions
          of smoke, fire, doors, exits, and zone labels. Does not move the
          agent but time still advances (fire spreads this step).
    """

    action: str = Field(..., description="Action type")
    direction: Optional[str] = Field(None, description="Cardinal direction for move")
    target_id: Optional[str] = Field(None, description="Door ID for door action")
    door_state: Optional[str] = Field(None, description="'open' or 'close' for door action")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PyreObservation(Observation):
    """Observation returned by the Pyre environment each step.

    The `narrative` field is the primary text the LLM reads. All other fields
    provide structured access to the same information for programmatic use.

    Inherited from Observation base: reward (float), done (bool), metadata (dict).
    """

    narrative: str = Field(default="", description="First-person narrative for the LLM agent")
    agent_evacuated: bool = Field(default=False, description="Whether agent has reached a safe exit")
    location_label: str = Field(default="", description="Current zone/room label")
    smoke_level: str = Field(default="none", description="none|light|moderate|heavy")
    fire_visible: bool = Field(default=False, description="Whether fire is in agent's sight")
    fire_direction: Optional[str] = Field(default=None, description="Direction of nearest fire")
    agent_health: float = Field(default=100.0, description="Agent health 0–100")
    health_status: str = Field(default="Good", description="Critical|Low|Moderate|Good")
    wind_dir: str = Field(default="CALM", description="Current wind direction")
    visible_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Objects in sight: [{id, type, relative_pos, state}]",
    )
    visible_people: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="NPCs in sight: [{id, relative_pos, state, distance}]",
    )
    blocked_exit_ids: List[str] = Field(
        default_factory=list,
        description="Exit IDs currently blocked by fire",
    )
    audible_signals: List[str] = Field(
        default_factory=list,
        description="Sounds the agent can hear",
    )
    elapsed_steps: int = Field(default=0, description="Steps elapsed in episode")
    last_action_feedback: str = Field(
        default="", description="Natural-language result of the previous action"
    )
    available_actions_hint: List[str] = Field(
        default_factory=list,
        description="Suggested action call strings for the current situation",
    )
    map_state: Optional[PyreMapState] = Field(
        default=None,
        description="Full numerical grid snapshot for UI rendering and external tooling",
    )


# ---------------------------------------------------------------------------
# State (server-side ground truth)
# ---------------------------------------------------------------------------

class PyreState(State):
    """Complete server-side ground truth for one episode.

    This is NOT sent to the agent — it drives the environment simulation.
    Inherited from State base: episode_id (str|None), step_count (int).
    """

    episode_id: Optional[str] = None
    step_count: int = 0

    # --- Map ---
    grid_w: int = 16
    grid_h: int = 16
    template_name: str = ""
    cell_grid: List[int] = Field(default_factory=list)
    fire_grid: List[float] = Field(default_factory=list)
    smoke_grid: List[float] = Field(default_factory=list)
    burn_timers: List[int] = Field(default_factory=list)
    exit_positions: List[List[int]] = Field(default_factory=list)
    door_registry: Dict[str, List[int]] = Field(default_factory=dict)
    zone_map: Dict[str, str] = Field(default_factory=dict)

    # --- Agent ---
    agent_x: int = 0
    agent_y: int = 0
    agent_alive: bool = True
    agent_evacuated: bool = False
    agent_health: float = 100.0

    # --- NPC tracking (updated each step) ---
    people_count: int = 0           # total NPCs spawned this episode
    people_evacuated: int = 0       # NPCs that reached a safe exit
    people_casualties: int = 0      # NPCs that became incapacitated
    stampede_events: int = 0        # total crush events triggered this episode

    # --- Episode fire config (randomized each episode) ---
    max_steps: int = 150
    fire_seed: int = 0
    fire_sources_count: int = 2
    fire_spread_rate: float = 0.25
    wind_dir: str = "CALM"
    humidity: float = 0.25

    # Visual-only furniture layer (fixed per floor plan template)
    furniture_map: Dict[str, str] = Field(default_factory=dict)
