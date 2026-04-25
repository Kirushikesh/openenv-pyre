"""
PyreEnvironment — single-agent crisis navigation environment.

Orchestrates:
  - Floor plan generation (floor_plan.py)
  - Fire/smoke dynamics with per-episode variability (fire_sim.py)
  - Narrative observation rendering (narrative.py)
  - Composite reward rubrics (rubrics.py)

Per-episode randomization (makes each episode unique):
  - Template selected in rotation
  - Number of fire ignition sources: 2–4
  - Fire spread rate: 0.15–0.40
  - Wind direction: random from 9 directions (N/NE/E/SE/S/SW/W/NW/CALM)
  - Humidity: 0.10–0.45
  - Agent spawn position: random from template spawn options
  - Fire start positions: random floor cells far from exits and agent

Done conditions:
  - Agent evacuated through an unblocked exit → success
  - Agent health reaches 0 (overwhelmed by smoke/fire) → failure
  - step_count >= max_steps → timeout
"""

import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

from ..models import PyreAction, PyreMapState, PyreObservation, PyreState
from .fire_sim import FireSim, FIRE_BURNING, smoke_level_label, WIND_DIRS
from .floor_plan import generate_episode, template_names
from .narrative import build_narrative_observation, compute_visible_cells
from .rubrics import make_per_step_rubrics, make_episode_end_rubrics

# Cell type constants
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

_CARDINAL_DELTA = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}

# Exit is blocked if fire intensity at exit cell exceeds this
EXIT_FIRE_THRESHOLD = 0.5

# Health damage per step (applied after each step)
DAMAGE_LIGHT_SMOKE = 0.5
DAMAGE_MODERATE_SMOKE = 2.0
DAMAGE_HEAVY_SMOKE = 5.0
DAMAGE_ON_FIRE = 10.0


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


class PyreEnvironment(Environment):
    """First-person crisis navigation environment — single agent.

    The agent is inside a burning building with partial observability.
    It must navigate to safety before its health depletes.
    Fire behaviour (spread rate, wind direction, ignition count) varies
    each episode, forcing the agent to reason from observations rather
    than memorising fixed patterns.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        max_steps: int = 150,
        base_seed: int = 42,
    ):
        super().__init__()

        self.max_steps = int(os.environ.get("PYRE_MAX_STEPS", max_steps))
        self.base_seed = int(os.environ.get("PYRE_SEED", base_seed))

        self._state: Optional[PyreState] = None
        self._fire_sim: Optional[FireSim] = None
        self._rng: Optional[random.Random] = None
        self._per_step_rubrics = make_per_step_rubrics()
        self._episode_rubrics = make_episode_end_rubrics()
        self._episode_counter = 0
        self._last_feedback = ""

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, **kwargs) -> PyreObservation:
        """Start a new episode with fresh randomized fire parameters."""
        fire_seed = seed if seed is not None else (self.base_seed + self._episode_counter * 37)
        self._episode_counter += 1
        self._rng = random.Random(fire_seed)

        # --- Randomize fire behaviour for this episode ---
        n_sources = self._rng.randint(2, 4)
        p_spread = round(self._rng.uniform(0.15, 0.40), 3)
        humidity = round(self._rng.uniform(0.10, 0.45), 3)
        wind_dir = self._rng.choice(list(WIND_DIRS.keys()))

        # --- Floor plan ---
        templates = template_names()
        template_name = templates[self._episode_counter % len(templates)]
        floor_plan, _, _, agent_start = generate_episode(
            template_name, npc_count=0, seed=fire_seed
        )

        w, h = floor_plan.w, floor_plan.h
        n_cells = w * h

        fire_grid = [0.0] * n_cells
        smoke_grid = [0.0] * n_cells
        burn_timers = [0] * n_cells

        # --- Place multiple fire sources ---
        floor_cells = [
            (x, y) for y in range(h) for x in range(w)
            if floor_plan.cell_grid[_idx(x, y, w)] == FLOOR
        ]
        fire_candidates = [
            pos for pos in floor_cells
            if all(
                _manhattan(pos[0], pos[1], ex[0], ex[1]) >= floor_plan.fire_min_exit_dist
                for ex in floor_plan.exit_positions
            )
            and _manhattan(pos[0], pos[1], agent_start[0], agent_start[1]) >= 4
        ]
        if not fire_candidates:
            fire_candidates = [
                pos for pos in floor_cells
                if pos != agent_start
                and pos not in [(e[0], e[1]) for e in floor_plan.exit_positions]
            ]

        n_sources = min(n_sources, len(fire_candidates))
        self._rng.shuffle(fire_candidates)
        fire_sources = fire_candidates[:n_sources]
        for fx, fy in fire_sources:
            fire_grid[_idx(fx, fy, w)] = 0.5

        # --- Door registry ---
        door_registry: Dict[str, List[int]] = {}
        for j, (dx, dy) in enumerate(floor_plan.door_positions):
            door_registry[f"door_{j + 1}"] = [dx, dy]

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
            agent_health=100.0,
            max_steps=self.max_steps,
            fire_seed=fire_seed,
            fire_sources_count=n_sources,
            fire_spread_rate=p_spread,
            wind_dir=wind_dir,
            humidity=humidity,
        )

        self._fire_sim = FireSim(
            w=w, h=h, rng=self._rng,
            p_spread=p_spread,
            wind_dir=wind_dir,
            humidity=humidity,
        )

        obs_data = build_narrative_observation(
            step_count=0,
            agent_x=agent_start[0],
            agent_y=agent_start[1],
            agent_alive=True,
            agent_evacuated=False,
            agent_health=100.0,
            cell_grid=floor_plan.cell_grid,
            fire_grid=fire_grid,
            smoke_grid=smoke_grid,
            exit_positions=[[ex[0], ex[1]] for ex in floor_plan.exit_positions],
            door_registry=door_registry,
            zone_map=floor_plan.zone_map,
            last_action_feedback=self._last_feedback,
            wind_dir=wind_dir,
            w=w, h=h,
        )
        obs_data["map_state"] = self._build_map_state(self._state)
        return PyreObservation(**obs_data)

    def step(self, action: PyreAction, **kwargs) -> PyreObservation:
        """Execute one action, advance simulation, return observation + reward."""
        st = self._state
        if st is None:
            st = self._make_default_state()

        prev_agent_x = st.agent_x
        prev_agent_y = st.agent_y
        prev_health = st.agent_health

        # --- Execute action ---
        feedback = self._execute_action(action, st)
        self._last_feedback = feedback

        # --- Check self-evacuation (must be unblocked exit) ---
        if st.agent_alive and not st.agent_evacuated:
            agent_cell = st.cell_grid[_idx(st.agent_x, st.agent_y, st.grid_w)]
            if agent_cell == EXIT:
                fire_at_exit = st.fire_grid[_idx(st.agent_x, st.agent_y, st.grid_w)]
                if fire_at_exit < EXIT_FIRE_THRESHOLD:
                    st.agent_evacuated = True
                    feedback = "You step through the exit and escape the building!"
                    self._last_feedback = feedback
                else:
                    feedback = "The exit is engulfed in flames — you can't get through!"
                    self._last_feedback = feedback

        # --- Advance fire simulation ---
        self._fire_sim.step(st.cell_grid, st.fire_grid, st.smoke_grid, st.burn_timers)

        # --- Apply health damage from smoke/fire ---
        health_damage = 0.0
        if st.agent_alive and not st.agent_evacuated:
            ai = _idx(st.agent_x, st.agent_y, st.grid_w)
            smoke = st.smoke_grid[ai]
            fire = st.fire_grid[ai]

            smoke_label = smoke_level_label(smoke)
            if smoke_label == "heavy":
                health_damage += DAMAGE_HEAVY_SMOKE
            elif smoke_label == "moderate":
                health_damage += DAMAGE_MODERATE_SMOKE
            elif smoke_label == "light":
                health_damage += DAMAGE_LIGHT_SMOKE

            if fire >= FIRE_BURNING:
                health_damage += DAMAGE_ON_FIRE

            st.agent_health = max(0.0, st.agent_health - health_damage)
            if st.agent_health <= 0:
                st.agent_alive = False
                self._last_feedback = "You collapse — overwhelmed by fire and smoke."

        st.step_count += 1

        # --- Done check ---
        done = self._check_done(st)

        # --- Reward ---
        reward = self._compute_reward(
            action=action.action,
            target_id=action.target_id,
            door_state=action.door_state,
            prev_agent_x=prev_agent_x,
            prev_agent_y=prev_agent_y,
            health_damage=health_damage,
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
            agent_health=st.agent_health,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            exit_positions=st.exit_positions,
            door_registry=st.door_registry,
            zone_map=st.zone_map,
            last_action_feedback=self._last_feedback,
            wind_dir=st.wind_dir,
            w=st.grid_w, h=st.grid_h,
        )
        obs_data["done"] = done
        obs_data["reward"] = reward
        obs_data["metadata"] = {
            "agent_health": st.agent_health,
            "step": st.step_count,
            "wind_dir": st.wind_dir,
            "fire_spread_rate": st.fire_spread_rate,
            "fire_sources": st.fire_sources_count,
            "humidity": st.humidity,
        }
        obs_data["map_state"] = self._build_map_state(st)
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
        elif act == "door":
            return self._action_door(action, st)
        elif act == "wait":
            return "You wait and listen to the building."
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

    def _action_door(self, action: PyreAction, st: PyreState) -> str:
        target_id = action.target_id
        door_state = (action.door_state or "").strip().lower()

        if not target_id:
            return "door requires a target_id (door ID)."
        if door_state not in ("open", "close"):
            return "door requires door_state='open' or door_state='close'."
        if target_id not in st.door_registry:
            return f"Door '{target_id}' not found."

        dx, dy = st.door_registry[target_id]
        if _manhattan(st.agent_x, st.agent_y, dx, dy) > 2:
            return f"Door '{target_id}' is too far away."

        ct = st.cell_grid[_idx(dx, dy, st.grid_w)]
        if ct not in (DOOR_OPEN, DOOR_CLOSED):
            return f"'{target_id}' is not a door."

        if door_state == "close":
            if ct == DOOR_CLOSED:
                return f"Door '{target_id}' is already closed."
            st.cell_grid[_idx(dx, dy, st.grid_w)] = DOOR_CLOSED
            return f"You close door '{target_id}'. It may slow the fire."
        else:
            if ct == DOOR_OPEN:
                return f"Door '{target_id}' is already open."
            st.cell_grid[_idx(dx, dy, st.grid_w)] = DOOR_OPEN
            return f"You open door '{target_id}'."

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
        return False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: str,
        target_id: Optional[str],
        door_state: Optional[str],
        prev_agent_x: int,
        prev_agent_y: int,
        health_damage: float,
        st: PyreState,
        done: bool,
    ) -> float:
        kwargs = dict(
            action=action,
            target_id=target_id,
            door_state=door_state,
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
            door_registry=st.door_registry,
            done=done,
            agent_evacuated=st.agent_evacuated,
            agent_alive=st.agent_alive,
            health_damage=health_damage,
            remaining_steps=max(0, st.max_steps - st.step_count),
        )

        total = 0.0
        for rubric in self._per_step_rubrics:
            total += rubric.score(**kwargs)
        if done:
            for rubric in self._episode_rubrics:
                total += rubric.score(**kwargs)

        return round(total, 4)

    # ------------------------------------------------------------------
    # Map state builder
    # ------------------------------------------------------------------

    def _build_map_state(self, st: PyreState) -> PyreMapState:
        """Assemble the full numerical grid snapshot for UI / visualization."""
        if st.agent_alive and not st.agent_evacuated:
            visible = compute_visible_cells(
                st.agent_x, st.agent_y,
                st.cell_grid, st.smoke_grid,
                st.grid_w, st.grid_h,
            )
            visible_cells = [[x, y] for x, y in sorted(visible)]
        else:
            visible_cells = []

        return PyreMapState(
            grid_w=st.grid_w,
            grid_h=st.grid_h,
            template_name=st.template_name,
            episode_id=st.episode_id or "",
            step_count=st.step_count,
            max_steps=st.max_steps,
            cell_grid=list(st.cell_grid),
            fire_grid=list(st.fire_grid),
            smoke_grid=list(st.smoke_grid),
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            agent_alive=st.agent_alive,
            agent_evacuated=st.agent_evacuated,
            agent_health=st.agent_health,
            visible_cells=visible_cells,
            exit_positions=list(st.exit_positions),
            door_registry=dict(st.door_registry),
            fire_spread_rate=st.fire_spread_rate,
            wind_dir=st.wind_dir,
            humidity=st.humidity,
        )

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
            agent_health=100.0,
            max_steps=self.max_steps,
            fire_seed=0,
            fire_sources_count=2,
            fire_spread_rate=0.25,
            wind_dir="CALM",
            humidity=0.25,
        )
