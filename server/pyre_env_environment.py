"""
PyreEnvironment — single-agent crisis navigation environment.

Orchestrates:
  - Floor plan generation (floor_plan.py)
  - Fire/smoke dynamics with per-episode variability (fire_sim.py)
  - Narrative observation rendering (narrative.py)
  - Composite reward rubrics (rubrics.py)

Per-episode randomization (makes each episode unique):
  - Template selected in rotation
  - Number of fire ignition sources: varies by difficulty
  - Fire spread rate: varies by difficulty
  - Wind direction: varies by difficulty
  - Humidity: varies by difficulty
  - Agent spawn position: random from template spawn options
  - Fire start positions: random floor cells far from exits and agent

Difficulty levels (set via the `difficulty` param in /reset):
  easy   — 1 source, slow spread, CALM wind, high humidity, 200 max steps
  medium — 2–4 sources, moderate spread, any wind, moderate humidity, 150 steps (default)
  hard   — 3–5 sources, fast spread, always windy, low humidity, 100 steps

Done conditions:
  - Agent evacuated through an unblocked exit → success
  - Agent health reaches 0 (overwhelmed by smoke/fire) → failure
  - step_count >= max_steps → timeout
"""

import os
import random
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import PyreAction, PyreMapState, PyreObservation, PyreState
except (ImportError, ModuleNotFoundError):
    from models import PyreAction, PyreMapState, PyreObservation, PyreState
from .fire_sim import FireSim, FIRE_BURNING, smoke_level_label, WIND_DIRS
from .floor_plan import generate_episode, generate_procedural_floor_plan, template_names
from .narrative import build_look_result, build_narrative_observation, compute_visible_cells
from .rubrics import make_per_step_rubrics, make_episode_end_rubrics, bfs_exit_dist, unblocked_exits, BFS_INF

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

# ---------------------------------------------------------------------------
# Difficulty presets — override fire randomisation ranges per episode
# ---------------------------------------------------------------------------
# Each preset is a dict of kwargs passed to reset(); ranges are (min, max) tuples.
_DIFFICULTY_PRESETS: Dict[str, Dict] = {
    "easy": {
        "n_sources_range": (1, 1),
        "p_spread_range": (0.10, 0.20),
        "humidity_range": (0.30, 0.50),
        "wind_choices": ["CALM"],
        "max_steps_override": 200,
        # Map layout: fixed 16×16 templates (stable topology aids early learning)
        "use_procedural": False,
        "grid_dims": (16, 16),
        "n_rooms_range": None,
        "template_pool": None,
    },
    "medium": {
        "n_sources_range": (2, 4),
        "p_spread_range": (0.15, 0.40),
        "humidity_range": (0.10, 0.45),
        "wind_choices": list(WIND_DIRS.keys()),
        "max_steps_override": None,  # use env default
        # Map layout: fixed 16×16 templates (same 3 buildings, fire varies)
        "use_procedural": False,
        "grid_dims": (16, 16),
        "n_rooms_range": None,
        "template_pool": None,
    },
    "hard_fixed": {
        "n_sources_range": (3, 5),
        "p_spread_range": (0.30, 0.55),
        "humidity_range": (0.05, 0.20),
        "wind_choices": [d for d in WIND_DIRS.keys() if d != "CALM"],
        "max_steps_override": 100,
        # Bridge stage: hard fire dynamics on a fixed topology before the
        # final jump to fully procedural buildings.
        "use_procedural": False,
        "grid_dims": (16, 16),
        "n_rooms_range": None,
        "template_pool": ["t_corridor"],
    },
    "hard": {
        "n_sources_range": (3, 5),
        "p_spread_range": (0.30, 0.55),
        "humidity_range": (0.05, 0.20),
        "wind_choices": [d for d in WIND_DIRS.keys() if d != "CALM"],
        "max_steps_override": 100,
        # Map layout: procedurally generated 20×24 building — every episode unique
        "use_procedural": True,
        "grid_dims": (20, 24),
        "n_rooms_range": (6, 10),
        "template_pool": None,
    },
}


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def _bfs_first_step_toward_exit(
    sx: int,
    sy: int,
    exits: List[List[int]],
    cell_grid: List[int],
    w: int,
    h: int,
) -> Optional[str]:
    """Return the first cardinal move on a shortest path to any reachable exit.

    Closed doors are treated as traversable for planning, matching the reward
    BFS used elsewhere. Returns None when no exit is reachable.
    """
    if not exits:
        return None

    exit_set = {(ex[0], ex[1]) for ex in exits}
    if (sx, sy) in exit_set:
        return None

    queue: deque = deque([(sx, sy, None)])
    visited = {(sx, sy)}
    moves = ((0, -1, "north"), (0, 1, "south"), (-1, 0, "west"), (1, 0, "east"))

    while queue:
        cx, cy, first_dir = queue.popleft()
        for dx, dy, dir_name in moves:
            nx, ny = cx + dx, cy + dy
            if not _in_bounds(nx, ny, w, h):
                continue
            if (nx, ny) in visited:
                continue
            ct = cell_grid[_idx(nx, ny, w)]
            if ct in (WALL, OBSTACLE):
                continue

            next_first = dir_name if first_dir is None else first_dir
            if (nx, ny) in exit_set:
                return next_first

            visited.add((nx, ny))
            queue.append((nx, ny, next_first))

    return None


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
        full_visibility: Optional[bool] = None,
    ):
        super().__init__()

        self.max_steps = int(os.environ.get("PYRE_MAX_STEPS", max_steps))
        self.base_seed = int(os.environ.get("PYRE_SEED", base_seed))
        if full_visibility is None:
            full_visibility = os.environ.get("PYRE_FULL_VISIBILITY", "1").strip().lower() not in {"0", "false", "no"}
        self.full_visibility = bool(full_visibility)

        self._state: Optional[PyreState] = None
        self._fire_sim: Optional[FireSim] = None
        self._rng: Optional[random.Random] = None
        self._per_step_rubrics = make_per_step_rubrics()
        self._episode_rubrics = make_episode_end_rubrics()
        self._episode_counter = 0
        self._last_feedback = ""

        # Episode-scoped reward tracking (reset each episode)
        self._visited_cells: set = set()
        self._min_exit_dist_reached: int = BFS_INF
        self._rewarded_doors: set = set()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, difficulty: str = "medium", **kwargs) -> PyreObservation:
        """Start a new episode with fresh randomized fire parameters.

        Args:
            seed:       Optional integer seed for full reproducibility.
            difficulty: "easy" | "medium" | "hard_fixed" | "hard" — scales fire
                        behaviour, topology variation, and episode length.
                        Pass via POST /reset body: {"difficulty": "easy"}
        """
        fire_seed = seed if seed is not None else (self.base_seed + self._episode_counter * 37)
        self._episode_counter += 1
        self._rng = random.Random(fire_seed)

        # --- Resolve difficulty preset ---
        preset = _DIFFICULTY_PRESETS.get(difficulty.lower(), _DIFFICULTY_PRESETS["medium"])
        n_min, n_max = preset["n_sources_range"]
        sp_min, sp_max = preset["p_spread_range"]
        hm_min, hm_max = preset["humidity_range"]
        wind_pool = preset["wind_choices"]
        effective_max_steps = preset["max_steps_override"] or self.max_steps

        # --- Randomize fire behaviour for this episode ---
        n_sources = self._rng.randint(n_min, n_max)
        p_spread = round(self._rng.uniform(sp_min, sp_max), 3)
        humidity = round(self._rng.uniform(hm_min, hm_max), 3)
        wind_dir = self._rng.choice(wind_pool)

        # --- Floor plan ---
        if preset["use_procedural"]:
            pw, ph = preset["grid_dims"]
            floor_plan = generate_procedural_floor_plan(
                w=pw, h=ph, rng=self._rng,
                n_rooms_range=preset["n_rooms_range"],
            )
            agent_start = self._rng.choice(floor_plan.agent_spawn_options)
            # Randomise some doors closed at episode start (same 30% rule as generate_episode)
            for dpos in floor_plan.door_positions:
                if self._rng.random() < 0.3:
                    floor_plan.cell_grid[_idx(dpos[0], dpos[1], pw)] = DOOR_CLOSED
            template_name = floor_plan.name
        else:
            templates = preset.get("template_pool") or template_names()
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
            # Start smoldering (0.1) so the agent has a few steps to observe
            # before fire reaches spreading intensity (>= 0.3).
            fire_grid[_idx(fx, fy, w)] = 0.1

        # --- Door registry ---
        door_registry: Dict[str, List[int]] = {}
        for j, (dx, dy) in enumerate(floor_plan.door_positions):
            door_registry[f"door_{j + 1}"] = [dx, dy]

        self._last_feedback = "Episode started. Assess your surroundings."
        self._difficulty = difficulty.lower()

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
            max_steps=effective_max_steps,
            fire_seed=fire_seed,
            fire_sources_count=n_sources,
            fire_spread_rate=p_spread,
            wind_dir=wind_dir,
            humidity=humidity,
        )

        # Reset episode-scoped reward tracking
        self._visited_cells = {(self._state.agent_x, self._state.agent_y)}
        self._min_exit_dist_reached = BFS_INF
        self._rewarded_doors = set()

        self._fire_sim = FireSim(
            w=w, h=h, rng=self._rng,
            p_spread=p_spread,
            wind_dir=wind_dir,
            humidity=humidity,
            fuel_map=floor_plan.fuel_map,
            ventilation_map=floor_plan.ventilation_map,
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
            visible_override=self._visible_set_for_state(self._state),
        )
        visible_set = self._visible_set_for_state(self._state)
        obs_data["metadata"] = self._build_observation_metadata(self._state, visible_set)
        obs_data["map_state"] = self._build_map_state(self._state, visible_set=visible_set)
        return PyreObservation(**obs_data)

    def _visible_set_for_state(self, st: PyreState) -> set:
        """Return the visibility set used for observations and encoded map_state."""
        if not st.agent_alive or st.agent_evacuated:
            return set()
        if self.full_visibility:
            return {(x, y) for y in range(st.grid_h) for x in range(st.grid_w)}
        return compute_visible_cells(
            st.agent_x, st.agent_y, st.cell_grid, st.smoke_grid, st.grid_w, st.grid_h,
        )

    def _build_observation_metadata(self, st: PyreState, visible_set: Optional[set] = None) -> Dict[str, Any]:
        """Metadata consumed by the trainer-side observation encoder."""
        if visible_set is None:
            visible_set = compute_visible_cells(
                st.agent_x, st.agent_y, st.cell_grid, st.smoke_grid, st.grid_w, st.grid_h,
            ) if st.agent_alive and not st.agent_evacuated else set()

        reachable_exits = unblocked_exits(st.exit_positions, st.fire_grid, st.grid_w)
        exits_for_dist = reachable_exits if reachable_exits else st.exit_positions
        nearest_exit_distance = bfs_exit_dist(
            st.agent_x, st.agent_y, exits_for_dist, st.cell_grid, st.grid_w, st.grid_h,
        )
        nearest_exit_direction = _bfs_first_step_toward_exit(
            st.agent_x, st.agent_y, exits_for_dist, st.cell_grid, st.grid_w, st.grid_h,
        )

        return {
            "agent_health": st.agent_health,
            "step": st.step_count,
            "wind_dir": st.wind_dir,
            "fire_spread_rate": st.fire_spread_rate,
            "fire_sources": st.fire_sources_count,
            "humidity": st.humidity,
            "difficulty": getattr(self, "_difficulty", "medium"),
            "nearest_exit_distance": nearest_exit_distance,
            "nearest_exit_direction": nearest_exit_direction,
            "reachable_exit_count": len(reachable_exits),
            "visible_cell_count": len(visible_set),
        }

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

        # --- Episode-scoped reward tracking ---
        # Detect first visit to current cell (before adding to visited set)
        is_new_cell = (st.agent_x, st.agent_y) not in self._visited_cells
        self._visited_cells.add((st.agent_x, st.agent_y))

        # Track closest approach to any exit for NearMissBonus
        _exits_reachable = unblocked_exits(st.exit_positions, st.fire_grid, st.grid_w)
        _exits = _exits_reachable if _exits_reachable else st.exit_positions
        _cur_dist = bfs_exit_dist(st.agent_x, st.agent_y, _exits, st.cell_grid, st.grid_w, st.grid_h)
        if _cur_dist < self._min_exit_dist_reached:
            self._min_exit_dist_reached = _cur_dist

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
            is_new_cell=is_new_cell,
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
            visible_override=self._visible_set_for_state(st),
        )
        obs_data["done"] = done
        obs_data["reward"] = reward

        # Compute visible cells once so both metadata and map_state can use the count
        _visible_set = self._visible_set_for_state(st)

        obs_data["metadata"] = self._build_observation_metadata(st, _visible_set)
        obs_data["map_state"] = self._build_map_state(st, visible_set=_visible_set)
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
        elif act == "look":
            return self._action_look(action, st)
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

    def _action_look(self, action: PyreAction, st: PyreState) -> str:
        direction = (action.direction or "").strip().lower()
        if not direction:
            return "look requires a direction: north, south, east, or west."
        return build_look_result(
            direction=direction,
            agent_x=st.agent_x,
            agent_y=st.agent_y,
            cell_grid=st.cell_grid,
            fire_grid=st.fire_grid,
            smoke_grid=st.smoke_grid,
            zone_map=st.zone_map,
            door_registry=st.door_registry,
            w=st.grid_w,
            h=st.grid_h,
        )

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
        is_new_cell: bool,
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
            agent_health=st.agent_health,
            health_damage=health_damage,
            remaining_steps=max(0, st.max_steps - st.step_count),
            is_new_cell=is_new_cell,
            min_exit_dist_reached=self._min_exit_dist_reached,
            rewarded_doors=self._rewarded_doors,
            reachable_exit_count=len(unblocked_exits(st.exit_positions, st.fire_grid, st.grid_w)),
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

    def _build_map_state(self, st: PyreState, visible_set: Optional[set] = None) -> PyreMapState:
        """Assemble the full numerical grid snapshot for UI / visualization.

        Args:
            visible_set: Pre-computed set of (x, y) visible cells. When provided
                         (e.g. from step()) the second compute_visible_cells call
                         is skipped. Pass None to compute fresh (used by reset()).
        """
        if st.agent_alive and not st.agent_evacuated:
            if visible_set is None:
                visible_set = self._visible_set_for_state(st)
            visible_cells = [[x, y] for x, y in sorted(visible_set)]
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
