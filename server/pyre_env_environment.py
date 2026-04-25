"""
PyreEnvironment — the main OpenEnv environment class for Pyre.

Orchestrates:
  - Floor plan generation (floor_plan.py)
  - Fire/smoke dynamics (fire_sim.py)
  - NPC state machine (npc_model.py)
  - Narrative observation rendering (narrative.py)
  - Composite reward rubrics (rubrics.py)

Episode lifecycle:
  reset() → sample a floor plan, place agent + NPCs + fire
  step(action) → execute action, advance simulation, compute reward, return observation
  state property → full server-side ground truth (PyreState)

Done conditions:
  - agent evacuated AND > 80% of starting NPCs have evacuated → success
  - agent incapacitated (dead) → failure
  - step_count >= max_steps → timeout, partial reward
  - all NPCs are either evacuated or casualties → forced end
"""

import copy
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

from ..models import PyreAction, PyreObservation, PyreState
from .fire_sim import FireSim, FIRE_BURNING, smoke_level_label
from .floor_plan import generate_episode, template_names
from .narrative import build_narrative_observation
from .npc_model import NPCModel, NPC_CALM, NPC_PANICKED, NPC_INJURED, NPC_INCAPACITATED
from .rubrics import (
    DoorTrapPenalty,
    make_per_step_rubrics,
    make_episode_end_rubrics,
)

# Cell type constants
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

_CARDINAL_DELTA = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


class PyreEnvironment(Environment):
    """First-person crisis navigation environment.

    The agent is inside a burning building. It must navigate to safety
    while coordinating NPC evacuations and managing fire spread through
    door control and broadcast instructions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        npc_count: int = 10,
        max_steps: int = 150,
        base_seed: int = 42,
    ):
        super().__init__()

        # Env var overrides
        self.npc_count = int(os.environ.get("PYRE_NPC_COUNT", npc_count))
        self.max_steps = int(os.environ.get("PYRE_MAX_STEPS", max_steps))
        self.base_seed = int(os.environ.get("PYRE_SEED", base_seed))

        self._state: Optional[PyreState] = None
        self._fire_sim: Optional[FireSim] = None
        self._npc_model: Optional[NPCModel] = None
        self._rng: Optional[random.Random] = None
        self._per_step_rubrics = make_per_step_rubrics()
        self._episode_rubrics = make_episode_end_rubrics()
        self._door_trap_rubric = DoorTrapPenalty()
        self._episode_counter = 0  # cycles through templates
        self._last_feedback = ""

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> PyreObservation:
        """Start a new episode: generate floor plan, place agent + NPCs + fire."""
        fire_seed = seed if seed is not None else (self.base_seed + self._episode_counter * 37)
        self._episode_counter += 1
        self._rng = random.Random(fire_seed)

        # Cycle through templates
        templates = template_names()
        template_name = templates[self._episode_counter % len(templates)]

        floor_plan, fire_start, npc_positions, agent_start = generate_episode(
            template_name, self.npc_count, seed=fire_seed
        )

        w, h = floor_plan.w, floor_plan.h
        n_cells = w * h

        # Initialize grids
        fire_grid = [0.0] * n_cells
        smoke_grid = [0.0] * n_cells
        burn_timers = [0] * n_cells

        # Place initial fire
        fire_grid[_idx(fire_start[0], fire_start[1], w)] = 0.5

        # Build NPC list
        npcs = []
        for idx_n, (nx, ny) in enumerate(npc_positions):
            npcs.append({
                "id": f"p_{idx_n + 1}",
                "x": nx,
                "y": ny,
                "state": NPC_CALM,
                "last_instruction_step": -1,
                "last_instruction_dir": None,
            })

        # Build door registry: id → [x, y]
        door_registry: Dict[str, List[int]] = {}
        for j, (dx, dy) in enumerate(floor_plan.door_positions):
            door_id = f"door_{j + 1}"
            door_registry[door_id] = [dx, dy]

        self._door_trap_rubric = DoorTrapPenalty()
        self._last_feedback = "Episode started. Assess your surroundings."

        self._state = PyreState.model_construct(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            grid_w=w,
            grid_h=h,
            template_name=template_name,
            cell_grid=floor_plan.cell_grid,
            fire_grid=fire_grid,
            smoke_grid=smoke_grid,
            burn_timers=burn_timers,
            exit_positions=[[ex[0], ex[1]] for ex in floor_plan.exit_positions],
            door_registry=door_registry,
            zone_map=floor_plan.zone_map,
            agent_x=agent_start[0],
            agent_y=agent_start[1],
            agent_alive=True,
            agent_evacuated=False,
            npcs=npcs,
            npcs_evacuated=0,
            npcs_casualties=0,
            stampede_events=0,
            doors_closed_by_agent=[],
            max_steps=self.max_steps,
            fire_seed=fire_seed,
            npc_count=len(npcs),
        )

        self._fire_sim = FireSim(w, h, self._rng)
        self._npc_model = NPCModel(w, h, floor_plan.exit_positions)

        obs_data = build_narrative_observation(
            step_count=0,
            agent_x=agent_start[0],
            agent_y=agent_start[1],
            agent_alive=True,
            agent_evacuated=False,
            cell_grid=floor_plan.cell_grid,
            fire_grid=fire_grid,
            smoke_grid=smoke_grid,
            npcs=npcs,
            exit_positions=[[ex[0], ex[1]] for ex in floor_plan.exit_positions],
            door_registry=door_registry,
            zone_map=floor_plan.zone_map,
            last_action_feedback=self._last_feedback,
            w=w, h=h,
        )
        return PyreObservation(**obs_data)

    def step(self, action: PyreAction, **kwargs) -> PyreObservation:  # type: ignore[override]
        """Execute one action, advance simulation, return observation + reward."""
        st = self._state
        if st is None:
            st = self._make_default_state()

        # Snapshot pre-step state for reward computation
        prev_agent_x = st.agent_x
        prev_agent_y = st.agent_y
        npcs_before = copy.deepcopy(st.npcs)
        prev_npcs_casualties = st.npcs_casualties
        prev_stampede_events = st.stampede_events

        # --- Execute action ---
        feedback = self._execute_action(action, st)
        self._last_feedback = feedback

        # Check agent self-evacuation
        if st.agent_alive and not st.agent_evacuated:
            agent_cell = st.cell_grid[_idx(st.agent_x, st.agent_y, st.grid_w)]
            if agent_cell == EXIT:
                st.agent_evacuated = True
                feedback = "You step through the exit and escape the building!"
                self._last_feedback = feedback

        # --- Advance fire simulation ---
        burned_out = self._fire_sim.step(
            st.cell_grid, st.fire_grid, st.smoke_grid, st.burn_timers
        )

        # Agent takes damage from fire/smoke
        if st.agent_alive and not st.agent_evacuated:
            ai = _idx(st.agent_x, st.agent_y, st.grid_w)
            if st.fire_grid[ai] > 0.7 or st.smoke_grid[ai] > 0.9:
                st.agent_alive = False
                self._last_feedback = "You are overcome by fire and smoke."

        # --- Advance NPCs ---
        if st.npcs:
            evacuated_ids, casualty_ids, stampede = self._npc_model.step_all(
                st.npcs,
                st.cell_grid,
                st.fire_grid,
                st.smoke_grid,
                (st.agent_x, st.agent_y),
                st.step_count,
                self._rng,
            )
            st.npcs_evacuated += len(evacuated_ids)
            st.npcs_casualties += len(casualty_ids)
            if stampede:
                st.stampede_events += 1
        else:
            casualty_ids = []
            stampede = False

        st.step_count += 1

        # --- Done check ---
        done = self._check_done(st)

        # --- Reward computation ---
        reward = self._compute_reward(
            action=action.action,
            target_id=action.target_id,
            prev_agent_x=prev_agent_x,
            prev_agent_y=prev_agent_y,
            npcs_before=npcs_before,
            new_casualty_ids=casualty_ids,
            st=st,
            done=done,
        )

        # --- Build observation ---
        obs_data = build_narrative_observation(
            step_count=st.step_count,
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            agent_alive=st.agent_alive,
            agent_evacuated=st.agent_evacuated,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            npcs=st.npcs,
            exit_positions=st.exit_positions,
            door_registry=st.door_registry,
            zone_map=st.zone_map,
            last_action_feedback=self._last_feedback,
            w=st.grid_w, h=st.grid_h,
        )
        obs_data["done"] = done
        obs_data["reward"] = reward
        obs_data["metadata"] = {
            "npcs_evacuated": st.npcs_evacuated,
            "npcs_casualties": st.npcs_casualties,
            "stampede_events": st.stampede_events,
            "step": st.step_count,
        }
        return PyreObservation(**obs_data)

    @property
    def state(self) -> PyreState:
        if self._state is None:
            self._state = self._make_default_state()
        return self._state

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: PyreAction, st: PyreState) -> str:
        act = action.action.strip().lower()

        if act == "move":
            return self._action_move(action, st)
        elif act == "instruct":
            return self._action_instruct(action, st)
        elif act == "close_door":
            return self._action_close_door(action, st)
        elif act == "open_door":
            return self._action_open_door(action, st)
        elif act == "broadcast":
            return self._action_broadcast(action, st)
        elif act == "wait":
            return "You wait, watching the situation."
        else:
            return f"Unknown action '{act}'. Nothing happened."

    def _action_move(self, action: PyreAction, st: PyreState) -> str:
        direction = (action.direction or "").lower()
        delta = _CARDINAL_DELTA.get(direction)
        if delta is None:
            return f"Invalid direction '{direction}'."

        nx, ny = st.agent_x + delta[0], st.agent_y + delta[1]

        if not _in_bounds(nx, ny, st.grid_w, st.grid_h):
            return "You walk into the outer wall — blocked."

        ct = st.cell_grid[_idx(nx, ny, st.grid_w)]
        if ct in (WALL, OBSTACLE):
            return "Blocked by wall or debris."
        if ct == DOOR_CLOSED:
            return f"The door to the {direction} is closed. Open it first."

        st.agent_x = nx
        st.agent_y = ny
        smoke = st.smoke_grid[_idx(nx, ny, st.grid_w)]
        fire = st.fire_grid[_idx(nx, ny, st.grid_w)]
        suffix = ""
        if smoke > 0.5:
            suffix = " The smoke is thick here."
        if fire > 0.1:
            suffix += " You feel intense heat."
        return f"You move {direction}.{suffix}"

    def _action_instruct(self, action: PyreAction, st: PyreState) -> str:
        target_id = action.target_id
        direction = (action.direction or "").lower()
        if not target_id:
            return "Instruct requires a target_id (person ID)."
        if direction not in _CARDINAL_DELTA:
            return f"Invalid direction '{direction}' for instruct."

        npc = next((n for n in st.npcs if n["id"] == target_id), None)
        if npc is None:
            return f"Person '{target_id}' not found or already evacuated."

        # Check within visibility radius (≈ 6 cells)
        dist = _manhattan(st.agent_x, st.agent_y, int(npc["x"]), int(npc["y"]))
        if dist > 6:
            return f"{target_id} is too far away to hear you."

        npc["last_instruction_dir"] = direction
        npc["last_instruction_step"] = st.step_count
        return f"You tell {target_id} to go {direction}."

    def _action_close_door(self, action: PyreAction, st: PyreState) -> str:
        target_id = action.target_id
        if not target_id:
            return "close_door requires a target_id (door ID)."
        if target_id not in st.door_registry:
            return f"Door '{target_id}' not found."

        dx, dy = st.door_registry[target_id]
        dist = _manhattan(st.agent_x, st.agent_y, dx, dy)
        if dist > 2:
            return f"Door '{target_id}' is too far away to reach."

        ct = st.cell_grid[_idx(dx, dy, st.grid_w)]
        if ct == DOOR_CLOSED:
            return f"Door '{target_id}' is already closed."
        if ct != DOOR_OPEN:
            return f"'{target_id}' is not a door."

        st.cell_grid[_idx(dx, dy, st.grid_w)] = DOOR_CLOSED
        if target_id not in st.doors_closed_by_agent:
            st.doors_closed_by_agent.append(target_id)

        # Record potential trap for DoorTrapPenalty
        self._door_trap_rubric.record_close(
            target_id, dx, dy, st.npcs, st.exit_positions
        )
        return f"You close door '{target_id}'. It may slow the fire."

    def _action_open_door(self, action: PyreAction, st: PyreState) -> str:
        target_id = action.target_id
        if not target_id:
            return "open_door requires a target_id (door ID)."
        if target_id not in st.door_registry:
            return f"Door '{target_id}' not found."

        dx, dy = st.door_registry[target_id]
        dist = _manhattan(st.agent_x, st.agent_y, dx, dy)
        if dist > 2:
            return f"Door '{target_id}' is too far away."

        ct = st.cell_grid[_idx(dx, dy, st.grid_w)]
        if ct == DOOR_OPEN:
            return f"Door '{target_id}' is already open."
        if ct != DOOR_CLOSED:
            return f"'{target_id}' is not a closeable door."

        st.cell_grid[_idx(dx, dy, st.grid_w)] = DOOR_OPEN
        return f"You open door '{target_id}'."

    def _action_broadcast(self, action: PyreAction, st: PyreState) -> str:
        zone = (action.zone or "").lower()
        category = (action.category or "").lower()
        if not zone or not category:
            return "broadcast requires both zone and category."

        # Map category to instruction direction
        cat_to_dir = {
            "evacuate_north": "north",
            "evacuate_south": "south",
            "evacuate_east": "east",
            "evacuate_west": "west",
            "use_exit_1": "south",  # heuristic
            "use_exit_2": "north",
            "stay_calm": None,
        }
        direction = cat_to_dir.get(category)

        # Apply to all NPCs whose zone matches
        count = 0
        for npc in st.npcs:
            nzone = st.zone_map.get(f"{int(npc['x'])},{int(npc['y'])}", "")
            if nzone == zone or zone == "all":
                if direction:
                    npc["last_instruction_dir"] = direction
                    npc["last_instruction_step"] = st.step_count
                    count += 1
                else:
                    # stay_calm: reduce panic probability slightly (mark with special step)
                    if npc["state"] == "panicked":
                        npc["last_instruction_dir"] = None
                        count += 1

        if count == 0:
            return f"Broadcast to zone '{zone}' — no one in range."
        return f"Broadcast to zone '{zone}' ({category}): {count} people may respond."

    # ------------------------------------------------------------------
    # Done check
    # ------------------------------------------------------------------

    def _check_done(self, st: PyreState) -> bool:
        if not st.agent_alive:
            return True
        if st.agent_evacuated:
            return True
        if st.step_count >= st.max_steps:
            return True
        # All NPCs resolved
        if len(st.npcs) == 0 and st.npc_count > 0:
            return True
        return False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: str,
        target_id: Optional[str],
        prev_agent_x: int,
        prev_agent_y: int,
        npcs_before: List[Dict],
        new_casualty_ids: List[str],
        st: PyreState,
        done: bool,
    ) -> float:
        kwargs = dict(
            action=action,
            target_id=target_id,
            prev_agent_x=prev_agent_x,
            prev_agent_y=prev_agent_y,
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            exit_positions=st.exit_positions,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            w=st.grid_w,
            h=st.grid_h,
            npcs_before=npcs_before,
            npcs_after=st.npcs,
            door_registry=st.door_registry,
            done=done,
            agent_evacuated=st.agent_evacuated,
            agent_alive=st.agent_alive,
            npcs_evacuated=st.npcs_evacuated,
            npcs_casualties=st.npcs_casualties,
            stampede_events=st.stampede_events,
            new_casualty_ids=new_casualty_ids,
        )

        total = 0.0
        for rubric in self._per_step_rubrics:
            total += rubric.score(**kwargs)
        if done:
            for rubric in self._episode_rubrics:
                total += rubric.score(**kwargs)

        # DoorTrapPenalty is stateful, handle separately
        total += self._door_trap_rubric.score(new_casualty_ids=new_casualty_ids)

        return round(total, 4)

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def _make_default_state(self) -> PyreState:
        return PyreState.model_construct(
            episode_id="",
            step_count=0,
            grid_w=16,
            grid_h=16,
            template_name="",
            cell_grid=[0] * 256,
            fire_grid=[0.0] * 256,
            smoke_grid=[0.0] * 256,
            burn_timers=[0] * 256,
            exit_positions=[],
            door_registry={},
            zone_map={},
            agent_x=0,
            agent_y=0,
            agent_alive=True,
            agent_evacuated=False,
            npcs=[],
            npcs_evacuated=0,
            npcs_casualties=0,
            stampede_events=0,
            doors_closed_by_agent=[],
            max_steps=self.max_steps,
            fire_seed=0,
            npc_count=0,
        )
