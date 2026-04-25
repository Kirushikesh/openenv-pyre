"""
Floor plan templates and episode generation for Pyre.

Three 16×16 hand-authored building templates:
  small_office  — two horizontal corridors + rooms, exits left/right
  open_plan     — open hall with pillar obstacles, exits at diagonal corners
  t_corridor    — T-shaped corridor network, three exits

Cell encoding:
  0 = floor       1 = wall        2 = door_open
  3 = door_closed 4 = exit        5 = obstacle
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    fire_min_exit_dist: int = 5            # fire ignition at least this far from any exit


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
#   Row  6: W F F F F F F F F F F F F F F W
#   Row  7: E F F F F F F F F F F F F F F E  ← exits
#   Row  8: W F F F F F F F F F F F F F F W
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
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 7  exits
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
        [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],  # 10 doors
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 11
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 12
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 13
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 14
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 15
    ]
    grid = [c for row in rows for c in row]

    exit_positions = [(0, 7), (15, 7)]
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
