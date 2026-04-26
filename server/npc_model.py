"""
Autonomous NPC behavior model for Pyre — Phase 1 (no agent communication).

State machine:
    Calm       → moves toward nearest exit via BFS; panics on fire/smoke exposure
    Panicked   → random walks away from fire/smoke; may be injured by crush events
    Injured    → stationary; cannot self-rescue
    Incapacitated → stationary; casualty (fire contact or prolonged heavy smoke)

Crush detection (called by environment after all NPCs step each turn):
    Density crush  — ≥ K panicked NPCs within radius 1 of any single cell
    Funnel crush   — ≥ FUNNEL_LIMIT panicked NPCs within radius 1 of an exit

Both types: each involved NPC has CRUSH_INJURY_PROB chance of transitioning Panicked→Injured.

Agent influence (indirect only, Phase 1):
    Closing a door makes it impassable for calm NPC pathfinding → redirects crowd flow.
    Opening a door unblocks a previously sealed route.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Cell type constants (mirrors fire_sim.py / floor_plan.py)
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

# Fire / smoke thresholds
FIRE_BURNING = 0.5
SMOKE_MODERATE = 0.4
SMOKE_HEAVY = 0.7

# Transition triggers
PANIC_SMOKE_STEPS = 2     # consecutive steps in moderate+ smoke → panic
INCAP_FIRE_STEPS = 3      # consecutive steps on burning cell → incapacitated
INCAP_SMOKE_STEPS = 10    # consecutive steps in heavy smoke → incapacitated

# Crush parameters
DENSITY_CRUSH_K = 3       # panicked NPCs within radius 1 of a cell to trigger density crush
FUNNEL_CRUSH_LIMIT = 2    # panicked NPCs within radius 1 of an exit to trigger funnel crush
CRUSH_INJURY_PROB = 0.60  # probability each NPC in a crush becomes injured

# Panicked NPC exit-seeking probability (panicked people still rush exits, just chaotically)
PANIC_EXIT_SEEK_PROB = 0.60


@dataclass
class NPC:
    """One civilian in the environment."""

    npc_id: str
    x: int
    y: int
    state: str = "calm"        # "calm" | "panicked" | "injured" | "incapacitated"

    # Exposure counters (reset on state-change)
    smoke_steps: int = 0       # consecutive steps in moderate+ smoke
    heavy_smoke_steps: int = 0 # consecutive steps in heavy smoke
    fire_steps: int = 0        # consecutive steps on a burning cell

    evacuated: bool = False


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _passable_for_npc(ct: int) -> bool:
    """Cell types an NPC can walk through autonomously (no door-opening ability)."""
    return ct in (FLOOR, DOOR_OPEN, EXIT)


# ---------------------------------------------------------------------------
# BFS: calm NPC pathfinding
# ---------------------------------------------------------------------------

def _bfs_next_step(
    sx: int, sy: int,
    exit_positions: List[Tuple[int, int]],
    cell_grid: List[int],
    fire_grid: List[float],
    w: int, h: int,
) -> Optional[Tuple[int, int]]:
    """Return the first step of the BFS path from (sx,sy) toward the nearest exit.

    Treats DOOR_CLOSED as impassable (calm NPCs cannot open doors).
    Returns None if no exit is reachable.
    """
    exit_set: Set[Tuple[int, int]] = set()
    for ex, ey in exit_positions:
        # Prefer unblocked exits; fall back to all exits if all are blocked
        if fire_grid[_idx(ex, ey, w)] < FIRE_BURNING:
            exit_set.add((ex, ey))
    if not exit_set:
        exit_set = {(ex, ey) for ex, ey in exit_positions}

    if (sx, sy) in exit_set:
        return None  # already at exit, step logic handles evacuation check

    # BFS — track parent for path reconstruction
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}
    queue: deque = deque()
    queue.append((sx, sy))

    target: Optional[Tuple[int, int]] = None
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in parent:
                continue
            ct = cell_grid[_idx(nx, ny, w)]
            if not _passable_for_npc(ct):
                continue
            parent[(nx, ny)] = (cx, cy)
            if (nx, ny) in exit_set:
                target = (nx, ny)
                break
            queue.append((nx, ny))
        if target:
            break

    if target is None:
        return None

    # Trace path back to first step
    node = target
    while parent[node] != (sx, sy):
        node = parent[node]  # type: ignore[assignment]
    return node


# ---------------------------------------------------------------------------
# Single NPC step
# ---------------------------------------------------------------------------

def step_npc(
    npc: NPC,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    exit_positions: List[Tuple[int, int]],
    w: int,
    h: int,
    rng: random.Random,
) -> None:
    """Advance one NPC by one timestep in-place.

    Called after the agent acts but before crush detection.
    Does nothing if the NPC is already evacuated, injured, or incapacitated.
    """
    if npc.evacuated or npc.state in ("injured", "incapacitated"):
        _update_counters_stationary(npc, cell_grid, fire_grid, smoke_grid, w)
        return

    # 1 — Update transition counters and state from current cell conditions
    _update_state_from_environment(npc, cell_grid, fire_grid, smoke_grid, w)

    if npc.state in ("injured", "incapacitated"):
        return

    # 2 — Move based on state
    if npc.state == "calm":
        _move_calm(npc, cell_grid, fire_grid, smoke_grid, exit_positions, w, h)
    elif npc.state == "panicked":
        _move_panicked(npc, cell_grid, fire_grid, smoke_grid, exit_positions, w, h, rng)

    # 3 — Check evacuation
    ct = cell_grid[_idx(npc.x, npc.y, w)]
    if ct == EXIT:
        fire_at = fire_grid[_idx(npc.x, npc.y, w)]
        if fire_at < FIRE_BURNING:
            npc.evacuated = True


def _update_state_from_environment(
    npc: NPC,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    w: int,
) -> None:
    """Update NPC state machine based on current cell fire/smoke conditions."""
    i = _idx(npc.x, npc.y, w)
    fire_val = fire_grid[i]
    smoke_val = smoke_grid[i]

    # Incapacitation via fire
    if fire_val >= FIRE_BURNING:
        npc.fire_steps += 1
        if npc.fire_steps >= INCAP_FIRE_STEPS:
            npc.state = "incapacitated"
            return
    else:
        npc.fire_steps = 0

    # Incapacitation via heavy smoke
    if smoke_val >= SMOKE_HEAVY:
        npc.heavy_smoke_steps += 1
        if npc.heavy_smoke_steps >= INCAP_SMOKE_STEPS:
            npc.state = "incapacitated"
            return
    else:
        npc.heavy_smoke_steps = 0

    # Panic from smoke (moderate+)
    if smoke_val >= SMOKE_MODERATE:
        npc.smoke_steps += 1
        if npc.smoke_steps >= PANIC_SMOKE_STEPS and npc.state == "calm":
            npc.state = "panicked"
            npc.smoke_steps = 0
            return
    else:
        npc.smoke_steps = 0

    # Panic from adjacent fire (calm NPC sees fire next to them)
    if npc.state == "calm" and _adjacent_fire(npc.x, npc.y, fire_grid, w, len(fire_grid) // w):
        npc.state = "panicked"


def _update_counters_stationary(
    npc: NPC,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    w: int,
) -> None:
    """Update exposure counters for stationary NPCs (injured/incapacitated)."""
    if npc.state == "incapacitated":
        return
    i = _idx(npc.x, npc.y, w)
    if fire_grid[i] >= FIRE_BURNING:
        npc.fire_steps += 1
        if npc.fire_steps >= INCAP_FIRE_STEPS:
            npc.state = "incapacitated"
    else:
        npc.fire_steps = 0

    if smoke_grid[i] >= SMOKE_HEAVY:
        npc.heavy_smoke_steps += 1
        if npc.heavy_smoke_steps >= INCAP_SMOKE_STEPS:
            npc.state = "incapacitated"
    else:
        npc.heavy_smoke_steps = 0


def _adjacent_fire(x: int, y: int, fire_grid: List[float], w: int, h: int) -> bool:
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            if fire_grid[_idx(nx, ny, w)] >= FIRE_BURNING:
                return True
    return False


def _move_calm(
    npc: NPC,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    exit_positions: List[Tuple[int, int]],
    w: int, h: int,
) -> None:
    """Move calm NPC one step toward nearest reachable exit via BFS."""
    next_step = _bfs_next_step(
        npc.x, npc.y,
        exit_positions,
        cell_grid, fire_grid,
        w, h,
    )
    if next_step is not None:
        npc.x, npc.y = next_step


def _move_panicked(
    npc: NPC,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    exit_positions: List[Tuple[int, int]],
    w: int, h: int,
    rng: random.Random,
) -> None:
    """Move panicked NPC: 60% rush toward nearest exit (chaotically), 40% flee from fire.

    Panicked people still try to escape — they just do it without coordination,
    which causes them to cluster at exits and trigger crush events.
    """
    # 60% chance: try one BFS step toward exit (same as calm but without avoiding smoke)
    if rng.random() < PANIC_EXIT_SEEK_PROB:
        next_step = _bfs_next_step(npc.x, npc.y, exit_positions, cell_grid, fire_grid, w, h)
        if next_step is not None:
            npc.x, npc.y = next_step
            return

    # 40% chance (or no exit reachable): flee away from highest fire/smoke
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def score(dx: int, dy: int) -> float:
        nx, ny = npc.x + dx, npc.y + dy
        if not (0 <= nx < w and 0 <= ny < h):
            return 999.0
        ct = cell_grid[_idx(nx, ny, w)]
        if not _passable_for_npc(ct):
            return 999.0
        i = _idx(nx, ny, w)
        return fire_grid[i] + smoke_grid[i]

    scored = sorted(directions, key=lambda d: (score(*d), rng.random()))
    candidates = [d for d in scored[:2] if score(*d) < 999.0]
    if not candidates:
        candidates = [d for d in scored if score(*d) < 999.0]
    if not candidates:
        return  # fully surrounded — stay put

    dx, dy = rng.choice(candidates)
    npc.x += dx
    npc.y += dy


# ---------------------------------------------------------------------------
# Crush detection
# ---------------------------------------------------------------------------

def detect_and_apply_crushes(
    npcs: List[NPC],
    exit_positions: List[Tuple[int, int]],
    rng: random.Random,
) -> int:
    """Detect density and funnel crush events; apply injuries. Returns crush count.

    Called after ALL NPCs have moved for the step.

    Crush types:
        Funnel crush — ≥ FUNNEL_CRUSH_LIMIT active panicked NPCs within radius 1
                       of any exit cell → crush at that exit.
        Density crush — ≥ DENSITY_CRUSH_K active panicked NPCs within radius 1
                        of any single NPC cell (excluding exit zones already caught
                        by funnel detection).
    """
    active = [n for n in npcs if not n.evacuated and n.state == "panicked"]
    if not active:
        return 0

    exit_set: Set[Tuple[int, int]] = {(ex, ey) for ex, ey in exit_positions}

    crush_events = 0
    injured_ids: Set[str] = set()

    # --- Funnel crush: panicked NPCs piling up near exits ---
    for ex, ey in exit_positions:
        near_exit = [
            n for n in active
            if abs(n.x - ex) + abs(n.y - ey) <= 1
        ]
        if len(near_exit) >= FUNNEL_CRUSH_LIMIT:
            crush_events += 1
            for npc in near_exit:
                if npc.npc_id not in injured_ids and rng.random() < CRUSH_INJURY_PROB:
                    npc.state = "injured"
                    injured_ids.add(npc.npc_id)

    # --- Density crush: general cluster of panicked NPCs ---
    for anchor in active:
        cluster = [
            n for n in active
            if abs(n.x - anchor.x) + abs(n.y - anchor.y) <= 1
            and (n.x, n.y) not in exit_set
        ]
        if len(cluster) >= DENSITY_CRUSH_K:
            crush_events += 1
            for npc in cluster:
                if npc.npc_id not in injured_ids and rng.random() < CRUSH_INJURY_PROB:
                    npc.state = "injured"
                    injured_ids.add(npc.npc_id)
            break  # count each density event once (avoid double-counting from every anchor)

    return crush_events


# ---------------------------------------------------------------------------
# Convenience helpers for the environment
# ---------------------------------------------------------------------------

def spawn_npcs(
    count: int,
    spawn_zones: List[Tuple[int, int]],
    agent_start: Tuple[int, int],
    rng: random.Random,
) -> List[NPC]:
    """Create `count` calm NPCs at random unique positions from spawn_zones."""
    available = [pos for pos in spawn_zones if pos != agent_start]
    rng.shuffle(available)
    chosen = available[:count]
    return [
        NPC(npc_id=f"person_{i + 1}", x=x, y=y)
        for i, (x, y) in enumerate(chosen)
    ]


def npc_summary(npcs: List[NPC]) -> Dict[str, int]:
    """Return counts by state for metadata/logging."""
    counts = {"calm": 0, "panicked": 0, "injured": 0, "incapacitated": 0, "evacuated": 0}
    for n in npcs:
        if n.evacuated:
            counts["evacuated"] += 1
        else:
            counts[n.state] = counts.get(n.state, 0) + 1
    return counts
