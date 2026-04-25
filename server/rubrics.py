"""
Composite reward rubrics for Pyre (single-agent).

Each rubric class exposes a score() method.
The environment composes them by calling each rubric each step.

Per-step rubrics:
  TimeStepPenalty          -0.01      constant time pressure
  ProgressReward           +0.1       agent moved closer to nearest unblocked exit
  DangerPenalty            -0.5       agent moved into smoke≥moderate or fire-adjacent cell
  HealthDrainPenalty       -0.02×dmg  proportional to health lost this step
  StrategicDoorBonus       +0.5       closed a door that meaningfully slows active fire

Episode-end rubrics:
  SelfSurviveBonus         +5.0       agent reached an open exit alive
  SelfDeathPenalty        -10.0       agent died (health depleted or fire/smoke)
  TimeBonus                +0.05×rem  reward for finishing quickly (remaining steps)
"""

from typing import Any, Dict, List, Optional

from .fire_sim import EXIT_BLOCKED_FIRE_THRESHOLD, FIRE_BURNING

EXIT = 4
WALL = 1
OBSTACLE = 5
DOOR_CLOSED = 3
SMOKE_MODERATE = 0.5


def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _nearest_exit_dist(x: int, y: int, exits: List[List[int]]) -> int:
    if not exits:
        return 9999
    return min(_manhattan(x, y, ex[0], ex[1]) for ex in exits)


def _unblocked_exits(exit_positions: List[List[int]], fire_grid: List[float], w: int) -> List[List[int]]:
    """Return exits that do not have significant fire on them."""
    return [
        ex for ex in exit_positions
        if fire_grid[ex[1] * w + ex[0]] < EXIT_BLOCKED_FIRE_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# Per-step rubrics
# ---------------------------------------------------------------------------

class TimeStepPenalty:
    """Small constant penalty per step to encourage urgency."""

    def score(self, **_) -> float:
        return -0.01


class ProgressReward:
    """Reward agent for moving strictly closer to the nearest unblocked exit."""

    def score(
        self,
        prev_agent_x: int, prev_agent_y: int,
        agent_x: int, agent_y: int,
        exit_positions: List[List[int]],
        fire_grid: List[float],
        w: int,
        action: str,
        **_,
    ) -> float:
        if action != "move":
            return 0.0
        exits = _unblocked_exits(exit_positions, fire_grid, w)
        if not exits:
            exits = exit_positions  # all blocked — still try to reward progress
        prev_dist = _nearest_exit_dist(prev_agent_x, prev_agent_y, exits)
        new_dist = _nearest_exit_dist(agent_x, agent_y, exits)
        return 0.1 if new_dist < prev_dist else 0.0


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
    """Bonus for closing a door adjacent to active fire — slows spread significantly."""

    def score(
        self,
        action: str,
        door_state: Optional[str],
        target_id: Optional[str],
        door_registry: Dict[str, List[int]],
        fire_grid: List[float],
        w: int, h: int,
        **_,
    ) -> float:
        if action != "door" or door_state != "close" or not target_id:
            return 0.0
        if target_id not in door_registry:
            return 0.0

        dx, dy = door_registry[target_id]
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = dx + ddx, dy + ddy
            if 0 <= nx < w and 0 <= ny < h:
                if fire_grid[ny * w + nx] >= FIRE_BURNING:
                    return 0.5
        return 0.0


# ---------------------------------------------------------------------------
# Episode-end rubrics
# ---------------------------------------------------------------------------

class SelfSurviveBonus:
    """Big bonus when agent evacuated alive."""

    def score(self, done: bool, agent_evacuated: bool, **_) -> float:
        return 5.0 if (done and agent_evacuated) else 0.0


class SelfDeathPenalty:
    """Big penalty when agent died (health depleted or overwhelmed by fire/smoke)."""

    def score(self, done: bool, agent_alive: bool, agent_evacuated: bool, **_) -> float:
        return -10.0 if (done and not agent_alive and not agent_evacuated) else 0.0


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
        DangerPenalty(),
        HealthDrainPenalty(),
        StrategicDoorBonus(),
    ]


def make_episode_end_rubrics():
    return [
        SelfSurviveBonus(),
        SelfDeathPenalty(),
        TimeBonus(),
    ]
