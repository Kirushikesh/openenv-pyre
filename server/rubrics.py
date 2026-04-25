"""
Composite reward rubrics for Pyre.

Each rubric class exposes a `score()` method and is stateless where possible.
The environment composes them by calling each rubric each step.

Per-step rubrics (called every step):
  TimeStepPenalty          -0.01   constant time pressure
  ProgressReward           +0.1    agent moved closer to nearest exit
  DangerPenalty            -0.5    agent moved into smoke≥moderate or fire-adjacent cell
  UsefulInstructionReward  +0.2    an instructed NPC followed and improved trajectory
  UselessInstructionPenalty -0.05  agent issued instruct/broadcast but no NPC complied
  StrategicDoorBonus       +0.5    closed a door that will meaningfully slow fire
  DoorTrapPenalty          -2.0    closed a door trapping an NPC who becomes a casualty

Episode-end rubrics (only fired when done=True):
  SelfSurviveBonus         +5.0    agent evacuated alive
  SelfDeathPenalty        -10.0    agent incapacitated
  EvacCountBonus           +1.0×N  per NPC evacuated
  CasualtyPenalty          -2.0×N  per NPC who became a casualty
  NoStampedeBonus          +3.0    episode ended without any stampede
  StampedePenalty          -1.5×N  per stampede event triggered
"""

from typing import Any, Dict, List, Optional, Tuple


def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _nearest_exit_dist(x: int, y: int, exits: List[List[int]]) -> int:
    if not exits:
        return 9999
    return min(_manhattan(x, y, ex[0], ex[1]) for ex in exits)


EXIT = 4
WALL = 1
OBSTACLE = 5
DOOR_CLOSED = 3
FIRE_BURNING = 0.3
SMOKE_MODERATE = 0.5


# ---------------------------------------------------------------------------
# Per-step rubrics
# ---------------------------------------------------------------------------

class TimeStepPenalty:
    """Small constant penalty per step to encourage urgency."""

    def score(self, **_) -> float:
        return -0.01


class ProgressReward:
    """Reward agent for moving strictly closer to the nearest exit."""

    def score(
        self,
        prev_agent_x: int, prev_agent_y: int,
        agent_x: int, agent_y: int,
        exit_positions: List[List[int]],
        action: str,
        **_,
    ) -> float:
        if action != "move":
            return 0.0
        prev_dist = _nearest_exit_dist(prev_agent_x, prev_agent_y, exit_positions)
        new_dist = _nearest_exit_dist(agent_x, agent_y, exit_positions)
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

        smoke = smoke_grid[i]
        if smoke >= SMOKE_MODERATE:
            return -0.5

        # Fire adjacent?
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = agent_x + dx, agent_y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if fire_grid[ny * w + nx] >= FIRE_BURNING:
                    return -0.5
        return 0.0


class UsefulInstructionReward:
    """Reward when an NPC that was just instructed actually followed and moved toward exit."""

    def score(
        self,
        action: str,
        target_id: Optional[str],
        npcs_before: List[Dict[str, Any]],
        npcs_after: List[Dict[str, Any]],
        exit_positions: List[List[int]],
        **_,
    ) -> float:
        if action not in ("instruct", "broadcast"):
            return 0.0
        if not target_id and action == "instruct":
            return 0.0

        # Check if any NPC improved their exit distance
        before_map = {n["id"]: n for n in npcs_before}
        for npc in npcs_after:
            nid = npc["id"]
            if nid not in before_map:
                continue
            old = before_map[nid]
            old_dist = _nearest_exit_dist(int(old["x"]), int(old["y"]), exit_positions)
            new_dist = _nearest_exit_dist(int(npc["x"]), int(npc["y"]), exit_positions)
            if new_dist < old_dist:
                return 0.2
        return 0.0


class UselessInstructionPenalty:
    """Small penalty when the agent issues an instruction that no one follows."""

    def score(
        self,
        action: str,
        target_id: Optional[str],
        npcs_before: List[Dict[str, Any]],
        npcs_after: List[Dict[str, Any]],
        exit_positions: List[List[int]],
        **_,
    ) -> float:
        if action not in ("instruct", "broadcast"):
            return 0.0

        before_map = {n["id"]: n for n in npcs_before}
        for npc in npcs_after:
            nid = npc["id"]
            if nid not in before_map:
                continue
            old = before_map[nid]
            old_dist = _nearest_exit_dist(int(old["x"]), int(old["y"]), exit_positions)
            new_dist = _nearest_exit_dist(int(npc["x"]), int(npc["y"]), exit_positions)
            if new_dist < old_dist:
                return 0.0  # at least one NPC moved — not useless
        return -0.05


class StrategicDoorBonus:
    """Bonus for closing a door that sits between a burning cell and floor cells."""

    def score(
        self,
        action: str,
        target_id: Optional[str],
        door_registry: Dict[str, List[int]],
        fire_grid: List[float],
        cell_grid: List[int],
        w: int, h: int,
        **_,
    ) -> float:
        if action != "close_door" or not target_id:
            return 0.0
        if target_id not in door_registry:
            return 0.0

        dx, dy = door_registry[target_id]

        # Check if any adjacent cell has active fire
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = dx + ddx, dy + ddy
            if 0 <= nx < w and 0 <= ny < h:
                if fire_grid[ny * w + nx] >= FIRE_BURNING:
                    return 0.5
        return 0.0


class DoorTrapPenalty:
    """Penalty when a door the agent closed traps an NPC who then becomes a casualty."""

    def __init__(self):
        self._trapped: Dict[str, List[str]] = {}  # door_id → [npc_ids on wrong side]

    def record_close(
        self,
        door_id: str,
        door_x: int, door_y: int,
        npcs: List[Dict[str, Any]],
        exit_positions: List[List[int]],
    ) -> None:
        """Call when agent closes a door to record which NPCs are on the far side."""
        if not exit_positions:
            return
        # NPCs are "trapped" if they're farther from all exits than the door itself
        door_dist = _nearest_exit_dist(door_x, door_y, exit_positions)
        trapped = [
            n["id"] for n in npcs
            if _nearest_exit_dist(int(n["x"]), int(n["y"]), exit_positions) > door_dist
        ]
        self._trapped[door_id] = trapped

    def score(
        self,
        new_casualty_ids: List[str],
        **_,
    ) -> float:
        penalty = 0.0
        for door_id, trapped_ids in self._trapped.items():
            for nid in new_casualty_ids:
                if nid in trapped_ids:
                    penalty -= 2.0
        return penalty


# ---------------------------------------------------------------------------
# Episode-end rubrics
# ---------------------------------------------------------------------------

class SelfSurviveBonus:
    """Big bonus if agent evacuated."""

    def score(self, done: bool, agent_evacuated: bool, **_) -> float:
        return 5.0 if (done and agent_evacuated) else 0.0


class SelfDeathPenalty:
    """Big penalty if agent died."""

    def score(self, done: bool, agent_alive: bool, **_) -> float:
        return -10.0 if (done and not agent_alive) else 0.0


class EvacCountBonus:
    """Bonus per NPC evacuated at episode end."""

    def score(self, done: bool, npcs_evacuated: int, **_) -> float:
        return 1.0 * npcs_evacuated if done else 0.0


class CasualtyPenalty:
    """Penalty per NPC casualty at episode end."""

    def score(self, done: bool, npcs_casualties: int, **_) -> float:
        return -2.0 * npcs_casualties if done else 0.0


class NoStampedeBonus:
    """Bonus if entire episode passed with no stampede."""

    def score(self, done: bool, stampede_events: int, **_) -> float:
        return 3.0 if (done and stampede_events == 0) else 0.0


class StampedePenalty:
    """Penalty per stampede event that occurred."""

    def score(self, done: bool, stampede_events: int, **_) -> float:
        return -1.5 * stampede_events if done else 0.0


# ---------------------------------------------------------------------------
# Convenience: all per-step rubrics and all episode-end rubrics
# ---------------------------------------------------------------------------

def make_per_step_rubrics():
    return [
        TimeStepPenalty(),
        ProgressReward(),
        DangerPenalty(),
        UsefulInstructionReward(),
        UselessInstructionPenalty(),
        StrategicDoorBonus(),
    ]


def make_episode_end_rubrics():
    return [
        SelfSurviveBonus(),
        SelfDeathPenalty(),
        EvacCountBonus(),
        CasualtyPenalty(),
        NoStampedeBonus(),
        StampedePenalty(),
    ]
