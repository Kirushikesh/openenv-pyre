"""
Floor plan templates and episode generation for Pyre.

Three 16×16 hand-authored building templates:
  small_office  — two horizontal corridors + rooms, exits left/right
  open_plan     — open hall with pillar obstacles, exits at diagonal corners
  t_corridor    — T-shaped corridor network, three exits

Cell encoding:
  0 = floor       1 = wall        2 = door_open
  3 = door_closed 4 = exit        5 = obstacle

fuel_map (per cell, float):
  Controls how fast fire ignites and intensifies in each cell.
  0.0 = no fuel (walls/obstacles)  1.0 = baseline  1.5 = high fuel (offices/rooms)

ventilation_map (per cell, float):
  Smoke decay rate per step for each cell. Higher = smoke clears faster.
  Open areas ventilate faster than enclosed rooms.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Cell type constants (mirrors fire_sim.py and models.py)
FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5


# ---------------------------------------------------------------------------
# FloorPlan dataclass
# ---------------------------------------------------------------------------

@dataclass
class FloorPlan:
    name: str
    cell_grid: List[int]                    # flattened H×W
    w: int
    h: int
    exit_positions: List[Tuple[int, int]]   # (x, y)
    door_positions: List[Tuple[int, int]]   # (x, y)
    spawn_zones: List[Tuple[int, int]]      # valid NPC spawn cells
    agent_spawn_options: List[Tuple[int, int]]
    zone_map: Dict[str, str]                # "{x},{y}" → zone_label
    fire_min_exit_dist: int = 5             # fire ignition at least this far from any exit
    fuel_map: List[float] = field(default_factory=list)         # fire fuel per cell
    ventilation_map: List[float] = field(default_factory=list)  # smoke decay per cell


# ---------------------------------------------------------------------------
# Fuel and ventilation helpers
# ---------------------------------------------------------------------------

# Fuel factor by zone label
_FUEL_BY_ZONE = {
    "north_offices":   1.5,   # paper, wooden furniture
    "south_offices":   1.5,
    "west_rooms":      1.5,
    "east_rooms":      1.5,
    "north_wing":      1.0,
    "south_wing":      1.0,
    "main_corridor":   1.0,
    "northwest_hall":  0.9,
    "northeast_hall":  0.9,
    "southwest_hall":  0.9,
    "southeast_hall":  0.9,
    "exit":            0.6,   # tile/concrete near exits
}
_FUEL_DEFAULT = 1.0
_FUEL_IMPASSABLE = 0.0  # walls and obstacles cannot burn

# Ventilation (smoke decay rate) by zone label
_VENT_BY_ZONE = {
    "main_corridor":   0.028,
    "north_wing":      0.025,
    "south_wing":      0.025,
    "northwest_hall":  0.050,  # large open plan — strong airflow
    "northeast_hall":  0.050,
    "southwest_hall":  0.050,
    "southeast_hall":  0.050,
    "north_offices":   0.010,  # enclosed rooms — smoke builds up
    "south_offices":   0.010,
    "west_rooms":      0.010,
    "east_rooms":      0.010,
    "exit":            0.040,  # exit gaps allow venting
}
_VENT_DEFAULT = 0.020
_VENT_IMPASSABLE = 0.0


def _build_fuel_and_ventilation(
    grid: List[int],
    zone_map: Dict[str, str],
    w: int,
    h: int,
) -> tuple[List[float], List[float]]:
    """Derive fuel_map and ventilation_map from zone labels and cell types."""
    fuel = []
    vent = []
    for y in range(h):
        for x in range(w):
            i = y * w + x
            ct = grid[i]
            if ct in (1, 5):   # wall or obstacle
                fuel.append(_FUEL_IMPASSABLE)
                vent.append(_VENT_IMPASSABLE)
            else:
                zone = zone_map.get(f"{x},{y}", "")
                fuel.append(_FUEL_BY_ZONE.get(zone, _FUEL_DEFAULT))
                vent.append(_VENT_BY_ZONE.get(zone, _VENT_DEFAULT))
    return fuel, vent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _cell_type(grid: List[int], x: int, y: int, w: int) -> int:
    return grid[_idx(x, y, w)]


# ---------------------------------------------------------------------------
# Template 1: small_office
#
# Layout (W=wall, F=floor, D=door_open, E=exit):
#
#   Row  0: W W W W W W W W W W W W W W W W
#   Row  1: W F F F W F F F W F F F W F F W
#   Row  2: W F F F W F F F W F F F W F F W
#   Row  3: W F F F W F F F W F F F W F F W
#   Row  4: W W D W W W D W W W D W W W D W  ← room→corridor doors
#   Row  5: W F F F F F F F F F F F F F F W  ← main corridor
#   Row  6: E F F F F F F F F F F F F F F W  ← west exit (staggered up)
#   Row  7: W F F F F F F F F F F F F F F W
#   Row  8: W F F F F F F F F F F F F F F E  ← east exit (staggered down)
#   Row  9: W F F F F F F F F F F F F F F W
#   Row 10: W W D W W W D W W W D W W W D W  ← room→corridor doors
#   Row 11: W F F F W F F F W F F F W F F W
#   Row 12: W F F F W F F F W F F F W F F W
#   Row 13: W F F F W F F F W F F F W F F W
#   Row 14: W F F F W F F F W F F F W F F W
#   Row 15: W W W W W W W W W W W W W W W W
# ---------------------------------------------------------------------------

def _make_small_office() -> FloorPlan:
    W, H = 16, 16
    rows = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 1
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 2
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 3
        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],  # 4  doors
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 5  corridor
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6  west exit
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 8  east exit
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],  # 10 doors
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 11
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 12
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 13
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 14
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 15
    ]
    grid = [c for row in rows for c in row]

    exit_positions = [(0, 6), (15, 8)]
    door_positions = [(2, 4), (6, 4), (10, 4), (14, 4),
                      (2, 10), (6, 10), (10, 10), (14, 10)]

    # Corridor cells (y=5-9, x=1-14) — agent spawns here
    corridor_cells = [(x, y) for y in range(5, 10) for x in range(1, 15)
                      if grid[_idx(x, y, W)] == 0]
    # Room cells for NPC spawning
    room_cells = [(x, y) for y in [1, 2, 3, 11, 12, 13, 14]
                  for x in range(1, 15)
                  if grid[_idx(x, y, W)] == 0]

    # Zone map: coarse labels
    zone_map: Dict[str, str] = {}
    for x in range(W):
        for y in range(H):
            ct = grid[_idx(x, y, W)]
            if ct == 0:
                if 5 <= y <= 9:
                    zone_map[f"{x},{y}"] = "main_corridor"
                elif y <= 4:
                    zone_map[f"{x},{y}"] = "north_offices"
                else:
                    zone_map[f"{x},{y}"] = "south_offices"
            elif ct == 4:
                zone_map[f"{x},{y}"] = "exit"

    fuel_map, ventilation_map = _build_fuel_and_ventilation(grid, zone_map, W, H)

    return FloorPlan(
        name="small_office",
        cell_grid=grid,
        w=W, h=H,
        exit_positions=exit_positions,
        door_positions=door_positions,
        spawn_zones=room_cells,
        agent_spawn_options=corridor_cells,
        zone_map=zone_map,
        fire_min_exit_dist=5,
        fuel_map=fuel_map,
        ventilation_map=ventilation_map,
    )


# ---------------------------------------------------------------------------
# Template 2: open_plan
#
# Layout:
#   Row  0: W W W W W W W W W W W W W W W W
#   Row  1: E F F F F F F F F F F F F F F W  ← exit at (0,1)
#   Row  2: W F F F F F F F F F F F F F F W
#   Row  3: W F F O O F F F F F O O F F F W  ← pillar obstacles
#   Row  4: W F F O O F F F F F O O F F F W
#   Row  5–10: open floor
#   Row 11: W F F O O F F F F F O O F F F W
#   Row 12: W F F O O F F F F F O O F F F W
#   Row 13: W F F F F F F F F F F F F F F W
#   Row 14: W F F F F F F F F F F F F F F W
#   Row 15: W W W W W W W W W W W W W W W E  ← exit at (15,15)
# ---------------------------------------------------------------------------

def _make_open_plan() -> FloorPlan:
    W, H = 16, 16
    rows = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1  exit at x=0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2
        [1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 1],  # 3  pillars
        [1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 1],  # 4
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 5
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
        [1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 1],  # 11 pillars
        [1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 1],  # 12
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 13
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],  # 15 exit at x=15
    ]
    grid = [c for row in rows for c in row]

    exit_positions = [(0, 1), (15, 15)]
    door_positions = []  # No internal doors in open plan

    floor_cells = [(x, y) for y in range(H) for x in range(W)
                   if grid[_idx(x, y, W)] == 0]

    zone_map: Dict[str, str] = {}
    for x in range(W):
        for y in range(H):
            ct = grid[_idx(x, y, W)]
            if ct == 0:
                if x <= 7 and y <= 7:
                    zone_map[f"{x},{y}"] = "northwest_hall"
                elif x > 7 and y <= 7:
                    zone_map[f"{x},{y}"] = "northeast_hall"
                elif x <= 7 and y > 7:
                    zone_map[f"{x},{y}"] = "southwest_hall"
                else:
                    zone_map[f"{x},{y}"] = "southeast_hall"
            elif ct == 4:
                zone_map[f"{x},{y}"] = "exit"

    fuel_map, ventilation_map = _build_fuel_and_ventilation(grid, zone_map, W, H)

    return FloorPlan(
        name="open_plan",
        cell_grid=grid,
        w=W, h=H,
        exit_positions=exit_positions,
        door_positions=door_positions,
        spawn_zones=floor_cells,
        agent_spawn_options=floor_cells,
        zone_map=zone_map,
        fire_min_exit_dist=4,
        fuel_map=fuel_map,
        ventilation_map=ventilation_map,
    )


# ---------------------------------------------------------------------------
# Template 3: t_corridor
#
# T-shaped layout: vertical stem (x=7, y=0-14) + horizontal bar (y=7, x=0-15)
# Side rooms off horizontal bar (y=8-12, left and right of stem):
#
#   Row  0: W W W W W W W E W W W W W W W W  ← exit at (7,0)
#   Row 1-6: vertical stem only (x=7)
#   Row  7: E F F F F F F F F F F F F F F E  ← horizontal bar + exits
#   Row  8: W F F W F F W F W F F W F F F W  ← rooms branch off bar
#   Row  9: W F F W F F W F W F F W F F F W
#   Row 10: W W D W W D W F W D W W W W D W  ← doors to stem
#   Row 11: W F F W F F W F W F F W F F F W
#   Row 12: W F F W F F W F W F F W F F F W
#   Row 13-14: stem only
#   Row 15: all walls
# ---------------------------------------------------------------------------

def _make_t_corridor() -> FloorPlan:
    W, H = 16, 16
    rows = [
        [1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1],  # 0  exit at x=7
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 1  stem
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 2
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 3
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 4
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 5
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 6
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 7  horizontal + exits
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # 8  side rooms
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # 9
        [1, 1, 2, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 1, 2, 1],  # 10 doors to stem
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # 11
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # 12
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 13 stem continues
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 14
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 15
    ]
    grid = [c for row in rows for c in row]

    exit_positions = [(7, 0), (0, 7), (15, 7)]
    door_positions = [(2, 10), (5, 10), (9, 10), (14, 10)]

    # Spawn zones: horizontal bar + side rooms
    bar_cells = [(x, 7) for x in range(1, 15) if grid[_idx(x, 7, W)] == 0]
    room_cells = [(x, y) for y in range(8, 13) for x in range(1, 15)
                  if grid[_idx(x, y, W)] == 0]
    stem_cells = [(7, y) for y in range(1, 15) if grid[_idx(7, y, W)] == 0]

    agent_spawn = bar_cells + [(7, y) for y in range(4, 8) if grid[_idx(7, y, W)] == 0]

    zone_map: Dict[str, str] = {}
    for x in range(W):
        for y in range(H):
            ct = grid[_idx(x, y, W)]
            if ct == 0:
                if y == 7:
                    zone_map[f"{x},{y}"] = "main_corridor"
                elif x == 7 and y < 7:
                    zone_map[f"{x},{y}"] = "north_wing"
                elif x == 7 and y > 7:
                    zone_map[f"{x},{y}"] = "south_wing"
                elif x < 7:
                    zone_map[f"{x},{y}"] = "west_rooms"
                else:
                    zone_map[f"{x},{y}"] = "east_rooms"
            elif ct == 4:
                zone_map[f"{x},{y}"] = "exit"

    fuel_map, ventilation_map = _build_fuel_and_ventilation(grid, zone_map, W, H)

    return FloorPlan(
        name="t_corridor",
        cell_grid=grid,
        w=W, h=H,
        exit_positions=exit_positions,
        door_positions=door_positions,
        spawn_zones=room_cells + bar_cells,
        agent_spawn_options=agent_spawn,
        zone_map=zone_map,
        fire_min_exit_dist=4,
        fuel_map=fuel_map,
        ventilation_map=ventilation_map,
    )


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: Optional[List[FloorPlan]] = None


def _get_templates() -> List[FloorPlan]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = [
            _make_small_office(),
            _make_open_plan(),
            _make_t_corridor(),
        ]
    return _TEMPLATES


def get_template(name: str) -> FloorPlan:
    for t in _get_templates():
        if t.name == name:
            return t
    raise ValueError(f"Unknown template: {name}")


def template_names() -> List[str]:
    return [t.name for t in _get_templates()]


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episode(
    template_name: str,
    npc_count: int,
    seed: int,
) -> Tuple[FloorPlan, Tuple[int, int], List[Tuple[int, int]], Tuple[int, int]]:
    """Generate a randomized episode from a template.

    Returns:
        (floor_plan, fire_start_xy, npc_positions, agent_start)
    """
    rng = random.Random(seed)
    fp = get_template(template_name)

    # Deep copy the cell_grid so templates are reusable
    cell_grid = fp.cell_grid[:]
    fp_copy = FloorPlan(
        name=fp.name,
        cell_grid=cell_grid,
        w=fp.w, h=fp.h,
        exit_positions=fp.exit_positions,
        door_positions=fp.door_positions,
        spawn_zones=fp.spawn_zones,
        agent_spawn_options=fp.agent_spawn_options,
        zone_map=fp.zone_map,
        fire_min_exit_dist=fp.fire_min_exit_dist,
        fuel_map=fp.fuel_map[:],
        ventilation_map=fp.ventilation_map[:],
    )

    # Agent start
    agent_start = rng.choice(fp.agent_spawn_options)

    # Randomize some doors to start closed (up to half)
    if fp.door_positions:
        for dpos in fp.door_positions:
            if rng.random() < 0.3:
                i = _idx(dpos[0], dpos[1], fp.w)
                fp_copy.cell_grid[i] = 3  # door_closed

    # NPC positions (from spawn_zones, no duplicates, not on agent start)
    available = [
        pos for pos in fp.spawn_zones
        if pos != agent_start
    ]
    rng.shuffle(available)
    npc_count = min(npc_count, len(available))
    npc_positions = available[:npc_count]

    # Fire start: random floor cell, far from all exits and agent
    floor_cells = [
        (x, y) for y in range(fp.h) for x in range(fp.w)
        if fp.cell_grid[_idx(x, y, fp.w)] == 0  # use original grid for fire candidates
    ]
    # Filter by min distance from exits
    candidates = [
        pos for pos in floor_cells
        if all(
            _manhattan(pos[0], pos[1], ex[0], ex[1]) >= fp.fire_min_exit_dist
            for ex in fp.exit_positions
        )
        and _manhattan(pos[0], pos[1], agent_start[0], agent_start[1]) >= 3
        and pos not in npc_positions
    ]
    if not candidates:
        # Fallback: any floor cell that isn't the agent or exit
        candidates = [
            pos for pos in floor_cells
            if pos != agent_start and pos not in [(e[0], e[1]) for e in fp.exit_positions]
        ]
    fire_start = rng.choice(candidates)

    return fp_copy, fire_start, npc_positions, agent_start


# ---------------------------------------------------------------------------
# Procedural floor plan generator
# ---------------------------------------------------------------------------
#
# Algorithm — Room-and-Corridor (4 phases):
#   1. Room placement   — random non-overlapping rectangles
#   2. MST corridors    — Prim-style minimum spanning tree connecting all rooms
#   3. Exit placement   — 2 exits on outer wall, maximally far apart
#   4. Zone + maps      — label cells, derive fuel/ventilation via existing helper
#
# Connectivity guard: BFS from a random agent spawn to confirm ≥1 exit is
# reachable. Up to 3 attempts; falls back to "small_office" if all fail.
# ---------------------------------------------------------------------------

# Room size ranges (interior dimensions, excluding surrounding walls)
_ROOM_MIN_W, _ROOM_MAX_W = 3, 5
_ROOM_MIN_H, _ROOM_MAX_H = 3, 4


def _rooms_collide(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> bool:
    """True if two interior rectangles are too close (need ≥1-cell wall gap)."""
    x1a, y1a, x2a, y2a = r1
    x1b, y1b, x2b, y2b = r2
    # Pad each room by 1 on all sides; collision if padded rects overlap.
    return not (x2a + 2 <= x1b or x2b + 2 <= x1a or y2a + 2 <= y1b or y2b + 2 <= y1a)


def _carve_hline(grid: List[int], x_start: int, x_end: int, y: int, w: int) -> None:
    for x in range(min(x_start, x_end), max(x_start, x_end) + 1):
        if 0 < x < w - 1:  # never overwrite outer boundary
            grid[y * w + x] = FLOOR


def _carve_vline(grid: List[int], y_start: int, y_end: int, x: int, w: int, h: int) -> None:
    for y in range(min(y_start, y_end), max(y_start, y_end) + 1):
        if 0 < y < h - 1:  # never overwrite outer boundary
            grid[y * w + x] = FLOOR


def _proc_bfs_reachable(
    sx: int, sy: int,
    exits: List[Tuple[int, int]],
    grid: List[int],
    w: int, h: int,
) -> bool:
    """Return True if any exit in `exits` is reachable from (sx, sy) via BFS."""
    exit_set = set(exits)
    if (sx, sy) in exit_set:
        return True
    visited = {(sx, sy)}
    queue = [(sx, sy)]
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in visited:
                continue
            ct = grid[ny * w + nx]
            if ct in (WALL, OBSTACLE):
                continue
            if (nx, ny) in exit_set:
                return True
            visited.add((nx, ny))
            queue.append((nx, ny))
    return False


def _try_generate_procedural(
    w: int,
    h: int,
    rng: random.Random,
    n_rooms_range: Tuple[int, int],
) -> Optional[FloorPlan]:
    """Single generation attempt. Returns FloorPlan on success, None on failure."""

    grid = [WALL] * (w * h)
    rooms: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2) interior rects

    # ------------------------------------------------------------------
    # Phase 1: Room placement
    # ------------------------------------------------------------------
    n_rooms = rng.randint(*n_rooms_range)

    for _ in range(n_rooms):
        for _attempt in range(200):
            rw = rng.randint(_ROOM_MIN_W, _ROOM_MAX_W)
            rh = rng.randint(_ROOM_MIN_H, _ROOM_MAX_H)
            # Keep 1-cell margin from outer boundary so exits can be placed cleanly
            x1 = rng.randint(2, w - rw - 3)
            y1 = rng.randint(2, h - rh - 3)
            x2 = x1 + rw - 1
            y2 = y1 + rh - 1
            new_room = (x1, y1, x2, y2)
            if not any(_rooms_collide(new_room, r) for r in rooms):
                rooms.append(new_room)
                for ry in range(y1, y2 + 1):
                    for rx in range(x1, x2 + 1):
                        grid[ry * w + rx] = FLOOR
                break

    if len(rooms) < 2:
        return None

    # ------------------------------------------------------------------
    # Phase 2: MST corridors (Prim-style, nearest-centre)
    # ------------------------------------------------------------------
    centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in rooms]
    room_cell_set: set = set()
    for x1, y1, x2, y2 in rooms:
        for ry in range(y1, y2 + 1):
            for rx in range(x1, x2 + 1):
                room_cell_set.add((rx, ry))

    connected = [0]
    unconnected = list(range(1, len(rooms)))

    while unconnected:
        best_dist, best_c, best_u = float("inf"), -1, -1
        for c in connected:
            for u in unconnected:
                cx_c, cy_c = centers[c]
                cx_u, cy_u = centers[u]
                d = abs(cx_c - cx_u) + abs(cy_c - cy_u)
                if d < best_dist:
                    best_dist, best_c, best_u = d, c, u

        cx_a, cy_a = centers[best_c]
        cx_b, cy_b = centers[best_u]

        # L-shaped corridor: random choice of which leg to draw first
        if rng.random() < 0.5:
            _carve_hline(grid, cx_a, cx_b, cy_a, w)
            _carve_vline(grid, cy_a, cy_b, cx_b, w, h)
        else:
            _carve_vline(grid, cy_a, cy_b, cx_a, w, h)
            _carve_hline(grid, cx_a, cx_b, cy_b, w)

        connected.append(best_u)
        unconnected.remove(best_u)

    # ------------------------------------------------------------------
    # Derive corridor cell set (FLOOR cells not inside any room)
    # ------------------------------------------------------------------
    corridor_cell_set: set = set()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if grid[y * w + x] == FLOOR and (x, y) not in room_cell_set:
                corridor_cell_set.add((x, y))

    # ------------------------------------------------------------------
    # Place DOOR_OPEN at room–corridor junctions (max 2 doors per room)
    # ------------------------------------------------------------------
    door_positions: List[Tuple[int, int]] = []
    doors_per_room = [0] * len(rooms)

    # Junction cells: corridor cells immediately adjacent to a room interior
    junction_candidates: List[Tuple[int, int]] = []
    for cx, cy in sorted(corridor_cell_set):  # sorted for determinism
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if (cx + dx, cy + dy) in room_cell_set:
                junction_candidates.append((cx, cy))
                break

    for jx, jy in junction_candidates:
        for i, (x1, y1, x2, y2) in enumerate(rooms):
            if doors_per_room[i] >= 2:
                continue
            # Is this junction adjacent to room i?
            adjacent = False
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                if x1 <= jx + dx <= x2 and y1 <= jy + dy <= y2:
                    adjacent = True
                    break
            if adjacent:
                grid[jy * w + jx] = DOOR_OPEN
                door_positions.append((jx, jy))
                doors_per_room[i] += 1
                break  # one room association per door cell

    # ------------------------------------------------------------------
    # Phase 3: Exit placement via dedicated tunnels
    #
    # Rather than relying on floor cells reaching the inner boundary, we
    # explicitly tunnel from the leftmost and rightmost floor cells to the
    # west (x=0) and east (x=w-1) outer walls. This is deterministic and
    # always succeeds as long as any floor cells were placed.
    # ------------------------------------------------------------------
    current_floor = [
        (x, y) for y in range(1, h - 1)
        for x in range(1, w - 1)
        if grid[y * w + x] == FLOOR
    ]
    if len(current_floor) < 2:
        return None

    # Exit 1 — west wall: use floor cell on leftmost column; carve west.
    left_cell = min(current_floor, key=lambda p: p[0])
    lx, ly = left_cell
    for cx in range(1, lx):           # carve any wall cells between x=1 and room
        grid[ly * w + cx] = FLOOR
    grid[ly * w + 0] = EXIT
    exit1: Tuple[int, int] = (0, ly)

    # Exit 2 — east wall: use floor cell on rightmost column; carve east.
    right_cell = max(current_floor, key=lambda p: p[0])
    rx, ry = right_cell
    for cx in range(rx + 1, w - 1):  # carve any wall cells between room and x=w-2
        grid[ry * w + cx] = FLOOR
    grid[ry * w + (w - 1)] = EXIT
    exit2: Tuple[int, int] = (w - 1, ry)

    exit_positions_list: List[Tuple[int, int]] = [exit1, exit2]

    # ------------------------------------------------------------------
    # Phase 4: Zone map, fuel, and ventilation
    #
    # Rebuild corridor_cell_set from the current grid so that exit tunnel
    # cells (carved in Phase 3) are included and labelled correctly.
    # ------------------------------------------------------------------
    corridor_cell_set = {
        (x, y)
        for y in range(1, h - 1)
        for x in range(1, w - 1)
        if grid[y * w + x] == FLOOR and (x, y) not in room_cell_set
    }

    zone_map: Dict[str, str] = {}

    # Rooms: use positional zone labels that match existing _FUEL_BY_ZONE / _VENT_BY_ZONE
    for i, (x1, y1, x2, y2) in enumerate(rooms):
        cx_room = (x1 + x2) // 2
        zone_label = "west_rooms" if cx_room < w // 2 else "east_rooms"
        for ry2 in range(y1, y2 + 1):
            for rx2 in range(x1, x2 + 1):
                if grid[ry2 * w + rx2] == FLOOR:
                    zone_map[f"{rx2},{ry2}"] = zone_label

    # Corridors (including exit tunnel cells) and doors
    for cx, cy in corridor_cell_set:
        if grid[cy * w + cx] in (FLOOR, DOOR_OPEN):
            zone_map[f"{cx},{cy}"] = "main_corridor"
    for dx, dy in door_positions:
        zone_map[f"{dx},{dy}"] = "main_corridor"

    # Exits
    for ex, ey in exit_positions_list:
        zone_map[f"{ex},{ey}"] = "exit"

    fuel_map, ventilation_map = _build_fuel_and_ventilation(grid, zone_map, w, h)

    # ------------------------------------------------------------------
    # Agent spawn options and connectivity guard
    # ------------------------------------------------------------------
    all_floor_cells = [
        (x, y) for y in range(h) for x in range(w)
        if grid[y * w + x] in (FLOOR, DOOR_OPEN)
    ]

    # Prefer corridor cells ≥4 cells from any exit (Manhattan)
    agent_spawn_options = [
        (x, y) for x, y in corridor_cell_set
        if grid[y * w + x] == FLOOR
        and all(
            abs(x - ex) + abs(y - ey) >= 4
            for ex, ey in exit_positions_list
        )
    ]
    if not agent_spawn_options:
        agent_spawn_options = [
            (x, y) for x, y in all_floor_cells
            if (x, y) not in exit_positions_list
        ]
    if not agent_spawn_options:
        return None

    # Connectivity guard: BFS from a sample spawn to verify exit is reachable
    test_spawn = rng.choice(agent_spawn_options)
    if not _proc_bfs_reachable(test_spawn[0], test_spawn[1], exit_positions_list, grid, w, h):
        return None

    return FloorPlan(
        name=f"procedural_{w}x{h}",
        cell_grid=grid,
        w=w, h=h,
        exit_positions=exit_positions_list,
        door_positions=door_positions,
        spawn_zones=all_floor_cells,
        agent_spawn_options=agent_spawn_options,
        zone_map=zone_map,
        fire_min_exit_dist=5,
        fuel_map=fuel_map,
        ventilation_map=ventilation_map,
    )


def generate_procedural_floor_plan(
    w: int,
    h: int,
    rng: random.Random,
    n_rooms_range: Tuple[int, int] = (6, 10),
) -> FloorPlan:
    """Generate a randomised floor plan procedurally.

    Tries up to 3 times with the given rng. Falls back to the hand-authored
    "small_office" template if all attempts fail (guarantees a valid plan is
    always returned).

    Args:
        w, h:           Grid dimensions (e.g. 20, 24 for hard difficulty).
        rng:            Seeded Random instance from the environment.
        n_rooms_range:  (min, max) number of rooms to attempt placing.
    """
    for _ in range(3):
        fp = _try_generate_procedural(w, h, rng, n_rooms_range)
        if fp is not None:
            return fp
    # Fallback: small_office is always valid and always connected
    return get_template("small_office")
