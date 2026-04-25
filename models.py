"""
Data models for the Pyre Environment.

Pyre is a first-person crisis navigation environment: an LLM agent navigates
a burning building, coordinates NPC evacuations, and manages fire spread under
partial observability. Internal state is a 2D grid; the agent receives a
structured textual narrative.

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
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PyreAction(Action):
    """Agent action for the Pyre environment.

    action:    "move" | "instruct" | "close_door" | "open_door" | "broadcast" | "wait"
    direction: "north"|"south"|"east"|"west"  — used by move and instruct
    target_id: door ID ("door_3") for close_door/open_door; person ID ("p_4") for instruct
    zone:      zone label for broadcast (e.g. "corridor", "north_wing")
    category:  broadcast category — one of:
               "evacuate_north" | "evacuate_south" | "evacuate_east" | "evacuate_west"
               | "use_exit_1" | "use_exit_2" | "stay_calm"
    """

    action: str = Field(..., description="Action type")
    direction: Optional[str] = Field(None, description="Cardinal direction")
    target_id: Optional[str] = Field(None, description="Door or person ID")
    zone: Optional[str] = Field(None, description="Zone label for broadcast")
    category: Optional[str] = Field(None, description="Broadcast category")


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
    location_label: str = Field(default="", description="Current zone/room label")
    smoke_level: str = Field(default="none", description="none|light|moderate|heavy")
    fire_visible: bool = Field(default=False, description="Whether fire is in agent's sight")
    fire_direction: Optional[str] = Field(default=None, description="Direction of nearest fire")
    visible_people: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="People in sight: [{id, relative_pos, state, last_seen_step}]",
    )
    visible_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Objects in sight: [{id, type, relative_pos, state}]",
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
    # Flattened 16×16 grid; cell encoding: 0=floor,1=wall,2=door_open,3=door_closed,4=exit,5=obstacle
    cell_grid: List[int] = Field(default_factory=list)
    fire_grid: List[float] = Field(default_factory=list)   # fire intensity 0.0–1.0
    smoke_grid: List[float] = Field(default_factory=list)  # smoke density 0.0–1.0
    burn_timers: List[int] = Field(default_factory=list)   # ticks a cell has been burning
    # Named exit positions as list of [x, y] pairs
    exit_positions: List[List[int]] = Field(default_factory=list)
    # Door registry: id → [x, y]
    door_registry: Dict[str, List[int]] = Field(default_factory=dict)
    # Zone labels: "{x},{y}" → zone_name
    zone_map: Dict[str, str] = Field(default_factory=dict)

    # --- Agent ---
    agent_x: int = 0
    agent_y: int = 0
    agent_alive: bool = True
    agent_evacuated: bool = False

    # --- NPCs ---
    # Each NPC: {id, x, y, state, last_instruction_step, last_instruction_dir}
    npcs: List[Dict[str, Any]] = Field(default_factory=list)
    npcs_evacuated: int = 0
    npcs_casualties: int = 0
    stampede_events: int = 0
    doors_closed_by_agent: List[str] = Field(default_factory=list)

    # --- Episode config ---
    max_steps: int = 150
    fire_seed: int = 0
    npc_count: int = 10
