"""
NPC behavior model for Pyre.

Each NPC has a state machine:
  CALM        → pathfinds to nearest exit; follows agent instructions (90% compliance)
  PANICKED    → moves randomly away from fire; follows instructions (50% compliance)
  INJURED     → stationary unless instructed within 2 steps (50% compliance)
  INCAPACITATED → cannot move; casualty if fire reaches them

State transitions:
  CALM → PANICKED:       sees adjacent fire OR own cell smoke > 0.5
  PANICKED → INJURED:    stampede event
  Any → INCAPACITATED:   own cell fire_intensity > 0.7 OR smoke > 0.9

Stampede: if ≥ 3 panicked NPCs within a 2-cell Manhattan radius in a narrow
corridor (adjacent walls make effective width ≤ 2), 30% of those NPCs become INJURED.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Cell type constants
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

# NPC states
NPC_CALM = "calm"
NPC_PANICKED = "panicked"
NPC_INJURED = "injured"
NPC_INCAPACITATED = "incapacitated"

# Compliance probabilities
CALM_INSTRUCTION_COMPLIANCE = 0.9
PANICKED_INSTRUCTION_COMPLIANCE = 0.5
INJURED_INSTRUCTION_COMPLIANCE = 0.5

STAMPEDE_THRESHOLD = 3      # panicked NPCs in radius 2 to trigger stampede
STAMPEDE_INJURY_PROB = 0.30 # each NPC in cluster becomes INJURED with this prob

# Passable cell types for BFS (NPC movement)
NPC_PASSABLE = {FLOOR, DOOR_OPEN, EXIT}
# Doors that NPCs can push through when panicked
PANICKED_PASSABLE = {FLOOR, DOOR_OPEN, DOOR_CLOSED, EXIT}


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


_CARDINAL = [(0, -1), (0, 1), (-1, 0), (1, 0)]

_DIR_TO_DELTA = {
    "north": (0, -1),
    "south": (0, 1),
    "west": (-1, 0),
    "east": (1, 0),
}


# ---------------------------------------------------------------------------
# BFS helpers
# ---------------------------------------------------------------------------

def bfs_next_step(
    sx: int,
    sy: int,
    targets: List[Tuple[int, int]],
    cell_grid: List[int],
    w: int,
    h: int,
    passable: Set[int],
) -> Optional[Tuple[int, int]]:
    """Return (dx,dy) for the first step of BFS toward nearest target.

    Returns None if no path exists.
    """
    if not targets:
        return None
    target_set = set(targets)
    if (sx, sy) in target_set:
        return (0, 0)

    visited: Set[Tuple[int, int]] = {(sx, sy)}
    # Queue stores (x, y, first_dx, first_dy)
    queue: deque = deque()
    for dx, dy in _CARDINAL:
        nx, ny = sx + dx, sy + dy
        if not _in_bounds(nx, ny, w, h):
            continue
        ct = cell_grid[_idx(nx, ny, w)]
        if ct in passable:
            queue.append((nx, ny, dx, dy))
            visited.add((nx, ny))

    while queue:
        x, y, fdx, fdy = queue.popleft()
        if (x, y) in target_set:
            return (fdx, fdy)
        for dx, dy in _CARDINAL:
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, w, h):
                continue
            if (nx, ny) in visited:
                continue
            ct = cell_grid[_idx(nx, ny, w)]
            if ct in passable:
                visited.add((nx, ny))
                queue.append((nx, ny, fdx, fdy))
    return None


def _direction_away_from_fire(
    x: int, y: int,
    fire_grid: List[float],
    cell_grid: List[int],
    w: int, h: int,
    rng: random.Random,
    passable: Set[int],
) -> Optional[Tuple[int, int]]:
    """Choose a step away from the highest nearby fire intensity."""
    best_delta = None
    best_score = float("inf")  # lower fire score = better to move there
    options = []
    for dx, dy in _CARDINAL:
        nx, ny = x + dx, y + dy
        if not _in_bounds(nx, ny, w, h):
            continue
        if cell_grid[_idx(nx, ny, w)] not in passable:
            continue
        fi = fire_grid[_idx(nx, ny, w)]
        options.append((fi, dx, dy))

    if not options:
        return None
    # Sort ascending fire intensity, pick lowest, with some randomness
    options.sort(key=lambda t: t[0])
    # With small probability try a different direction (panic randomness)
    if rng.random() < 0.3 and len(options) > 1:
        return options[rng.randint(0, min(2, len(options) - 1))][1:]
    return options[0][1:]


# ---------------------------------------------------------------------------
# NPCModel
# ---------------------------------------------------------------------------

class NPCModel:
    """Manages all NPC behavior for one episode."""

    def __init__(
        self,
        w: int,
        h: int,
        exit_positions: List[Tuple[int, int]],
    ):
        self.w = w
        self.h = h
        self.exit_positions = exit_positions

    def step_all(
        self,
        npcs: List[Dict[str, Any]],   # mutated in-place
        cell_grid: List[int],
        fire_grid: List[float],
        smoke_grid: List[float],
        agent_pos: Tuple[int, int],
        step_count: int,
        rng: random.Random,
    ) -> Tuple[List[str], List[str], bool]:
        """Advance all NPCs by one step.

        Returns:
            (evacuated_ids, casualty_ids, stampede_occurred)
        """
        evacuated_ids: List[str] = []
        casualty_ids: List[str] = []
        to_remove: List[str] = []
        stampede_occurred = False

        active = [n for n in npcs if n["state"] not in (NPC_INCAPACITATED,)]

        for npc in active:
            nid = npc["id"]
            x, y = int(npc["x"]), int(npc["y"])
            state = npc["state"]
            i = _idx(x, y, self.w)

            # -- State transitions (checked before movement) --
            fire_here = fire_grid[i]
            smoke_here = smoke_grid[i]

            if fire_here > 0.7 or smoke_here > 0.9:
                npc["state"] = NPC_INCAPACITATED
                casualty_ids.append(nid)
                to_remove.append(nid)
                continue

            if state == NPC_CALM:
                # Check adjacent fire visibility
                adj_fire = any(
                    fire_grid[_idx(x + dx, y + dy, self.w)] > 0.1
                    for dx, dy in _CARDINAL
                    if _in_bounds(x + dx, y + dy, self.w, self.h)
                )
                if adj_fire or smoke_here > 0.5:
                    npc["state"] = NPC_PANICKED
                    state = NPC_PANICKED

            # -- Movement --
            if state == NPC_INCAPACITATED:
                continue

            delta = self._decide_move(npc, cell_grid, fire_grid, smoke_grid, step_count, rng)

            if delta is not None:
                nx, ny = x + int(delta[0]), y + int(delta[1])
                if _in_bounds(nx, ny, self.w, self.h):
                    npc["x"] = nx
                    npc["y"] = ny
                    # Check evacuation
                    ct = cell_grid[_idx(nx, ny, self.w)]
                    if ct == EXIT:
                        evacuated_ids.append(nid)
                        to_remove.append(nid)

        # Remove evacuated/casualties
        remove_set = set(to_remove)
        npcs[:] = [n for n in npcs if n["id"] not in remove_set]

        # -- Stampede detection --
        panicked = [n for n in npcs if n["state"] == NPC_PANICKED]
        if len(panicked) >= STAMPEDE_THRESHOLD:
            stampede_occurred = self._check_stampede(npcs, cell_grid, rng)

        return evacuated_ids, casualty_ids, stampede_occurred

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _decide_move(
        self,
        npc: Dict[str, Any],
        cell_grid: List[int],
        fire_grid: List[float],
        smoke_grid: List[float],
        step_count: int,
        rng: random.Random,
    ) -> Optional[Tuple[int, int]]:
        state = npc["state"]
        x, y = int(npc["x"]), int(npc["y"])

        # Check if following an active instruction
        instruction_active = (
            npc.get("last_instruction_dir") is not None
            and step_count - npc.get("last_instruction_step", -99) <= 2
        )

        # Decide whether to follow instruction
        if instruction_active:
            compliance = (
                CALM_INSTRUCTION_COMPLIANCE if state == NPC_CALM
                else PANICKED_INSTRUCTION_COMPLIANCE if state == NPC_PANICKED
                else INJURED_INSTRUCTION_COMPLIANCE
            )
            if rng.random() < compliance:
                direction = npc["last_instruction_dir"]
                delta = _DIR_TO_DELTA.get(direction)
                if delta:
                    nx, ny = x + delta[0], y + delta[1]
                    if _in_bounds(nx, ny, self.w, self.h):
                        passable = NPC_PASSABLE if state != NPC_PANICKED else PANICKED_PASSABLE
                        if cell_grid[_idx(nx, ny, self.w)] in passable:
                            return delta
                        # Instruction blocked — fall through to default behavior

        if state == NPC_INJURED:
            return None  # injured NPCs don't move without instruction

        if state == NPC_CALM:
            passable = NPC_PASSABLE
            delta = bfs_next_step(x, y, self.exit_positions, cell_grid, self.w, self.h, passable)
            return delta

        if state == NPC_PANICKED:
            passable = PANICKED_PASSABLE
            delta = _direction_away_from_fire(x, y, fire_grid, cell_grid, self.w, self.h, rng, passable)
            if delta is None:
                # Fallback: BFS to exit
                delta = bfs_next_step(x, y, self.exit_positions, cell_grid, self.w, self.h, passable)
            return delta

        return None

    def _check_stampede(
        self,
        npcs: List[Dict[str, Any]],
        cell_grid: List[int],
        rng: random.Random,
    ) -> bool:
        """Detect and apply stampede in narrow corridors."""
        stampede_triggered = False
        panicked = [n for n in npcs if n["state"] == NPC_PANICKED]

        for npc in panicked:
            cx, cy = int(npc["x"]), int(npc["y"])

            # Count panicked NPCs within Manhattan radius 2
            nearby_panicked = [
                n for n in panicked
                if abs(int(n["x"]) - cx) + abs(int(n["y"]) - cy) <= 2
            ]

            if len(nearby_panicked) < STAMPEDE_THRESHOLD:
                continue

            # Check if in narrow corridor: any direction has walls on both sides
            in_narrow = self._is_narrow_corridor(cx, cy, cell_grid)
            if not in_narrow:
                continue

            stampede_triggered = True
            for victim in nearby_panicked:
                if rng.random() < STAMPEDE_INJURY_PROB:
                    victim["state"] = NPC_INJURED

        return stampede_triggered

    def _is_narrow_corridor(
        self, x: int, y: int, cell_grid: List[int]
    ) -> bool:
        """Check if (x,y) is in a narrow corridor (width ≤ 2 in any axis)."""
        w, h = self.w, self.h
        blocking = (WALL, OBSTACLE)

        # Check east-west axis: walls north and south within 1 cell?
        north_blocked = (
            not _in_bounds(x, y - 1, w, h)
            or cell_grid[_idx(x, y - 1, w)] in blocking
        )
        south_blocked = (
            not _in_bounds(x, y + 1, w, h)
            or cell_grid[_idx(x, y + 1, w)] in blocking
        )
        if north_blocked or south_blocked:
            return True

        # Check north-south axis: walls east and west within 1 cell?
        west_blocked = (
            not _in_bounds(x - 1, y, w, h)
            or cell_grid[_idx(x - 1, y, w)] in blocking
        )
        east_blocked = (
            not _in_bounds(x + 1, y, w, h)
            or cell_grid[_idx(x + 1, y, w)] in blocking
        )
        if west_blocked or east_blocked:
            return True

        return False
