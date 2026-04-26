"""
Composite reward rubrics for Pyre (single-agent).

Each rubric class exposes a score() method.
The environment composes them by calling each rubric each step.

Per-step rubrics:
  TimeStepPenalty            -0.01      constant time pressure
  ProgressReward             +0.1       agent moved closer to nearest unblocked exit (BFS distance)
  ProgressRegressionPenalty  -0.05      agent moved farther from nearest exit (symmetric gradient)
  SafeProgressBonus          +0.05      stacks on ProgressReward when progress made through smoke-free cell
  DangerPenalty              -0.5       agent moved into smoke≥moderate or fire-adjacent cell
  HealthDrainPenalty         -0.02×dmg  proportional to health lost this step
  StrategicDoorBonus         +0.5       closed a door adjacent to active fire (once per door per episode)
  ExplorationBonus           +0.02      first visit to a cell in this episode

Episode-end rubrics:
  SelfSurviveBonus         +5.0              agent reached an open exit alive
  HealthSurvivalBonus      +1.5×(hp/100)     graduated bonus for evacuating with more health
  SelfDeathPenalty        -10.0              agent died (health depleted or fire/smoke)
  TimeoutPenalty           -5.0              agent ran out of steps while still alive (lighter than death)
  NearMissBonus            0→+3.0            partial credit on failure based on closest approach to exit
  TimeBonus                +0.05×rem         reward for finishing quickly (remaining steps)
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set

from .fire_sim import EXIT_BLOCKED_FIRE_THRESHOLD, FIRE_BURNING

EXIT = 4
WALL = 1
OBSTACLE = 5
DOOR_CLOSED = 3
SMOKE_MODERATE = 0.5

BFS_INF = 9999
_BFS_INF = BFS_INF  # keep private alias for internal use


def unblocked_exits(exit_positions: List[List[int]], fire_grid: List[float], w: int) -> List[List[int]]:
    """Return exits that do not have significant fire on them."""
    return [
        ex for ex in exit_positions
        if fire_grid[ex[1] * w + ex[0]] < EXIT_BLOCKED_FIRE_THRESHOLD
    ]


# Legacy private alias so existing internal callers don't break.
_unblocked_exits = unblocked_exits


def bfs_exit_dist(
    x: int,
    y: int,
    exits: List[List[int]],
    cell_grid: List[int],
    w: int,
    h: int,
) -> int:
    """BFS traversal distance from (x, y) to the nearest reachable exit.

    Walls (1) and obstacles (5) block movement. Closed doors (3) are treated
    as passable — the agent can open them en route. Returns BFS_INF when no
    exit is reachable (all paths wall-blocked).
    """
    if not exits:
        return BFS_INF

    exit_set = {(ex[0], ex[1]) for ex in exits}
    if (x, y) in exit_set:
        return 0

    visited = {(x, y)}
    queue: deque = deque()
    queue.append((x, y, 0))

    while queue:
        cx, cy, dist = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in visited:
                continue
            ct = cell_grid[ny * w + nx]
            if ct in (WALL, OBSTACLE):
                continue
            if (nx, ny) in exit_set:
                return dist + 1
            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return BFS_INF


# Legacy private alias so existing internal callers don't break.
_bfs_exit_dist = bfs_exit_dist


# ---------------------------------------------------------------------------
# Per-step rubrics
# ---------------------------------------------------------------------------

class TimeStepPenalty:
    """Small constant penalty per step to encourage urgency."""

    def score(self, **_) -> float:
        return -0.01


class ProgressReward:
    """Reward agent for moving strictly closer to the nearest unblocked exit.

    Uses BFS traversal distance (respects walls and obstacles) instead of
    Manhattan distance, so only genuine navigational progress is rewarded.
    """

    def score(
        self,
        prev_agent_x: int, prev_agent_y: int,
        agent_x: int, agent_y: int,
        exit_positions: List[List[int]],
        fire_grid: List[float],
        cell_grid: List[int],
        w: int, h: int,
        action: str,
        **_,
    ) -> float:
        if action != "move":
            return 0.0
        exits = _unblocked_exits(exit_positions, fire_grid, w)
        if not exits:
            exits = exit_positions  # all blocked — still try to reward progress
        prev_dist = _bfs_exit_dist(prev_agent_x, prev_agent_y, exits, cell_grid, w, h)
        new_dist = _bfs_exit_dist(agent_x, agent_y, exits, cell_grid, w, h)
        return 0.15 if new_dist < prev_dist else 0.0


class DangerPenalty:
    """Penalise moving into a dangerous cell (smoke ≥ moderate or fire adjacent)."""

    def score(
        self,
        agent_x: int, agent_y: int,
        action: str,
        cell_grid: List[int],
        fire_grid: List[float],
        smoke_grid: List[float],
        w: int, h: int,
        **_,
    ) -> float:
        if action != "move":
            return 0.0

        i = agent_y * w + agent_x
        if i < 0 or i >= len(smoke_grid):
            return 0.0

        if smoke_grid[i] >= SMOKE_MODERATE:
            return -0.5

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = agent_x + dx, agent_y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if fire_grid[ny * w + nx] >= FIRE_BURNING:
                    return -0.5
        return 0.0


class HealthDrainPenalty:
    """Penalty proportional to health damage taken this step from smoke/fire.

    Moderate smoke (~2 dmg/step) → -0.04/step
    Heavy smoke   (~5 dmg/step) → -0.10/step
    On fire       (~10 dmg/step) → -0.20/step
    """

    def score(self, health_damage: float, **_) -> float:
        return -0.02 * health_damage


class StrategicDoorBonus:
    """Bonus for closing a door adjacent to active fire — slows spread significantly.

    Each door can only earn this bonus once per episode. The environment passes
    a mutable `rewarded_doors` set; the rubric checks and updates it to prevent
    the agent from farming +0.5 by repeatedly opening and closing the same door.
    """

    def score(
        self,
        action: str,
        door_state: Optional[str],
        target_id: Optional[str],
        door_registry: Dict[str, List[int]],
        fire_grid: List[float],
        rewarded_doors: Set[str],
        w: int, h: int,
        **_,
    ) -> float:
        if action != "door" or door_state != "close" or not target_id:
            return 0.0
        if target_id not in door_registry:
            return 0.0
        if target_id in rewarded_doors:
            return 0.0

        dx, dy = door_registry[target_id]
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = dx + ddx, dy + ddy
            if 0 <= nx < w and 0 <= ny < h:
                if fire_grid[ny * w + nx] >= FIRE_BURNING:
                    rewarded_doors.add(target_id)
                    return 0.5
        return 0.0


class ProgressRegressionPenalty:
    """Penalise moving farther from the nearest unblocked exit.

    Symmetric counterpart to ProgressReward: the agent gets +0.1 for progress
    and –0.05 for regression, creating a clean two-sided gradient.
    """

    def score(
        self,
        prev_agent_x: int, prev_agent_y: int,
        agent_x: int, agent_y: int,
        exit_positions: List[List[int]],
        fire_grid: List[float],
        cell_grid: List[int],
        w: int, h: int,
        action: str,
        **_,
    ) -> float:
        if action != "move":
            return 0.0
        exits = unblocked_exits(exit_positions, fire_grid, w)
        if not exits:
            exits = exit_positions
        prev_dist = bfs_exit_dist(prev_agent_x, prev_agent_y, exits, cell_grid, w, h)
        new_dist = bfs_exit_dist(agent_x, agent_y, exits, cell_grid, w, h)
        return -0.05 if new_dist > prev_dist else 0.0


class SafeProgressBonus:
    """Extra reward for making exit-progress through a smoke-free cell.

    Stacks on top of ProgressReward to teach the agent to prefer safe routes
    when multiple paths lead equally close to the exit.
    """

    def score(
        self,
        prev_agent_x: int, prev_agent_y: int,
        agent_x: int, agent_y: int,
        exit_positions: List[List[int]],
        fire_grid: List[float],
        smoke_grid: List[float],
        cell_grid: List[int],
        w: int, h: int,
        action: str,
        **_,
    ) -> float:
        if action != "move":
            return 0.0
        exits = unblocked_exits(exit_positions, fire_grid, w)
        if not exits:
            exits = exit_positions
        prev_dist = bfs_exit_dist(prev_agent_x, prev_agent_y, exits, cell_grid, w, h)
        new_dist = bfs_exit_dist(agent_x, agent_y, exits, cell_grid, w, h)
        if new_dist >= prev_dist:
            return 0.0
        i = agent_y * w + agent_x
        return 0.05 if (0 <= i < len(smoke_grid) and smoke_grid[i] < SMOKE_MODERATE) else 0.0


class ExplorationBonus:
    """Small bonus for visiting a cell for the first time in the episode.

    Prevents the agent standing still or looping to avoid DangerPenalty.
    The environment tracks visited cells and passes `is_new_cell`.
    """

    def score(self, action: str, is_new_cell: bool, **_) -> float:
        return 0.02 if (action == "move" and is_new_cell) else 0.0


# ---------------------------------------------------------------------------
# Episode-end rubrics
# ---------------------------------------------------------------------------

class SelfSurviveBonus:
    """Big bonus when agent evacuated alive."""

    def score(self, done: bool, agent_evacuated: bool, **_) -> float:
        return 5.0 if (done and agent_evacuated) else 0.0


class HealthSurvivalBonus:
    """Graduated bonus for evacuating with remaining health.

    Rewards finding the *safest* route, not just any route. An agent that
    evacuates at 95 HP earns ~+1.43; one that barely survives at 5 HP earns
    ~+0.075. Range: 0 → +1.5.
    """

    def score(self, done: bool, agent_evacuated: bool, agent_health: float, **_) -> float:
        return 1.5 * (agent_health / 100.0) if (done and agent_evacuated) else 0.0


class SelfDeathPenalty:
    """Big penalty when agent died (health depleted or overwhelmed by fire/smoke)."""

    def score(self, done: bool, agent_alive: bool, agent_evacuated: bool, **_) -> float:
        return -10.0 if (done and not agent_alive and not agent_evacuated) else 0.0


class TimeoutPenalty:
    """Penalty for running out of steps while still alive without evacuating.

    Maintains the ordering: success > timeout > death.

    If exits were reachable (not all blocked by fire), the penalty scales with
    remaining health — a healthy agent that timed out demonstrably could have
    evacuated, so it receives a stronger signal (-5 to -8).

    If all exits were fire-blocked, the agent had no path out regardless of
    health, so a flat -5 is applied (not the agent's fault).
    """

    def score(
        self,
        done: bool,
        agent_alive: bool,
        agent_evacuated: bool,
        agent_health: float,
        reachable_exit_count: int,
        **_,
    ) -> float:
        if not (done and agent_alive and not agent_evacuated):
            return 0.0
        if reachable_exit_count == 0:
            return -5.0  # exits were fire-blocked — flat penalty, not the agent's fault
        # Agent had open exits but still timed out — scale by how healthy it was.
        # hp=100 → -8.0,  hp=50 → -6.5,  hp=10 → -5.3
        return -5.0 - 3.0 * (agent_health / 100.0)


class NearMissBonus:
    """Graduated partial credit on DEATH based on closest exit approach.

    Fixes hard-mode reward collapse: when all episodes end in death the flat
    SelfDeathPenalty creates zero gradient. This rubric differentiates "almost
    made it" from "never got close" using the minimum BFS distance reached.

    Formula: max(0.0, 3.0 – 0.5 × min_exit_dist_reached)
      dist=0 → +3.0  (at exit but died — edge case)
      dist=1 → +2.5
      dist=3 → +1.5
      dist=6 → 0.0   (no credit beyond 6 cells away)

    Only fires on DEATH — not on timeout. A timed-out agent was alive and
    had the opportunity to act; softening the TimeoutPenalty here would
    undermine the health-scaled signal from TimeoutPenalty.
    """

    def score(
        self,
        done: bool,
        agent_alive: bool,
        agent_evacuated: bool,
        min_exit_dist_reached: int,
        **_,
    ) -> float:
        # Only award on death (not alive, not evacuated)
        if not done or agent_evacuated or agent_alive:
            return 0.0
        return max(0.0, 3.0 - 0.5 * min_exit_dist_reached)


class TimeBonus:
    """Bonus for escaping quickly — rewards remaining steps when agent evacuates."""

    def score(self, done: bool, agent_evacuated: bool, remaining_steps: int, **_) -> float:
        return 0.05 * remaining_steps if (done and agent_evacuated) else 0.0


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_per_step_rubrics():
    return [
        TimeStepPenalty(),
        ProgressReward(),
        ProgressRegressionPenalty(),
        SafeProgressBonus(),
        DangerPenalty(),
        HealthDrainPenalty(),
        StrategicDoorBonus(),
        ExplorationBonus(),
    ]


def make_episode_end_rubrics():
    return [
        SelfSurviveBonus(),
        HealthSurvivalBonus(),
        SelfDeathPenalty(),
        TimeoutPenalty(),
        NearMissBonus(),
        TimeBonus(),
    ]
