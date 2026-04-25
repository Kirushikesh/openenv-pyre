"""
Observation rendering for Pyre (single-agent).

Converts raw server state into:
  - A first-person narrative string (the LLM's primary input)
  - Structured fields in PyreObservation

Visibility rules:
  - Base radius: 5 (Manhattan distance)
  - Moderate smoke in agent's cell: radius 3
  - Heavy smoke in agent's cell: radius 2
  - Walls block flood-fill propagation

Health status labels:
  80–100 : Good
  50–79  : Moderate
  25–49  : Low
  0–24   : Critical
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .fire_sim import smoke_level_label, FIRE_BURNING, EXIT_BLOCKED_FIRE_THRESHOLD

FLOOR = 0
WALL = 1
DOOR_OPEN = 2
DOOR_CLOSED = 3
EXIT = 4
OBSTACLE = 5

_CARDINAL = [(0, -1, "north"), (0, 1, "south"), (-1, 0, "west"), (1, 0, "east")]
_DELTA_TO_DIR = {(0, -1): "north", (0, 1): "south", (-1, 0): "west", (1, 0): "east"}


def _idx(x: int, y: int, w: int) -> int:
    return y * w + x


def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _health_label(health: float) -> str:
    if health >= 80:
        return "Good"
    if health >= 50:
        return "Moderate"
    if health >= 25:
        return "Low"
    return "Critical"


# ---------------------------------------------------------------------------
# Visibility computation
# ---------------------------------------------------------------------------

def compute_visible_cells(
    ax: int, ay: int,
    cell_grid: List[int],
    smoke_grid: List[float],
    w: int, h: int,
) -> Set[Tuple[int, int]]:
    """BFS flood-fill from agent; walls block propagation."""
    agent_smoke = smoke_grid[_idx(ax, ay, w)]
    label = smoke_level_label(agent_smoke)

    if label == "heavy":
        radius = 2
    elif label == "moderate":
        radius = 3
    else:
        radius = 5

    visible: Set[Tuple[int, int]] = {(ax, ay)}
    queue = [(ax, ay, 0)]

    while queue:
        x, y, dist = queue.pop(0)
        if dist >= radius:
            continue
        for dx, dy, _ in _CARDINAL:
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, w, h):
                continue
            if (nx, ny) in visible:
                continue
            ct = cell_grid[_idx(nx, ny, w)]
            if ct == WALL:
                continue
            visible.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return visible


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_narrative_observation(
    step_count: int,
    agent_x: int,
    agent_y: int,
    agent_alive: bool,
    agent_evacuated: bool,
    agent_health: float,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    exit_positions: List[List[int]],
    door_registry: Dict[str, List[int]],
    zone_map: Dict[str, str],
    last_action_feedback: str,
    wind_dir: str,
    w: int,
    h: int,
) -> Dict[str, Any]:
    """Build the full observation dict (matches PyreObservation fields)."""
    if agent_evacuated:
        return _terminal_obs(step_count, last_action_feedback,
                             narrative="You have safely evacuated the building.",
                             agent_evacuated=True)

    if not agent_alive:
        return _terminal_obs(step_count, last_action_feedback,
                             narrative="You have been overcome by fire and smoke.",
                             agent_evacuated=False)

    visible = compute_visible_cells(agent_x, agent_y, cell_grid, smoke_grid, w, h)

    # --- Agent cell conditions ---
    agent_smoke = smoke_grid[_idx(agent_x, agent_y, w)]
    smoke_label = smoke_level_label(agent_smoke)
    health_label = _health_label(agent_health)

    # --- Fire visibility ---
    fire_visible = False
    fire_dir: Optional[str] = None
    nearest_fire_dist = 999
    for vx, vy in visible:
        if (vx, vy) == (agent_x, agent_y):
            continue
        if fire_grid[_idx(vx, vy, w)] >= FIRE_BURNING:
            fire_visible = True
            d = _manhattan(agent_x, agent_y, vx, vy)
            if d < nearest_fire_dist:
                nearest_fire_dist = d
                dx = vx - agent_x
                dy = vy - agent_y
                if abs(dx) >= abs(dy):
                    fire_dir = "east" if dx > 0 else "west"
                else:
                    fire_dir = "south" if dy > 0 else "north"

    # --- Visible objects (doors and exits) ---
    visible_objects: List[Dict[str, Any]] = []
    door_pos_to_id = {(v[0], v[1]): k for k, v in door_registry.items()}
    blocked_exit_ids: List[str] = []

    for vx, vy in visible:
        ct = cell_grid[_idx(vx, vy, w)]
        rel = _relative_pos_str(agent_x, agent_y, vx, vy)

        if ct in (DOOR_OPEN, DOOR_CLOSED):
            door_id = door_pos_to_id.get((vx, vy), f"door_{vx}_{vy}")
            door_state = "open" if ct == DOOR_OPEN else "closed"
            if fire_grid[_idx(vx, vy, w)] > 0.1:
                door_state += " (hot)"
            visible_objects.append({
                "id": door_id, "type": "door",
                "relative_pos": rel, "state": door_state,
            })

        elif ct == EXIT:
            fire_at_exit = fire_grid[_idx(vx, vy, w)]
            exit_id = f"exit_{vx}_{vy}"
            if fire_at_exit >= EXIT_BLOCKED_FIRE_THRESHOLD:
                exit_state = "BLOCKED by fire"
                blocked_exit_ids.append(exit_id)
            else:
                exit_state = "open"
            visible_objects.append({
                "id": exit_id, "type": "exit",
                "relative_pos": rel, "state": exit_state,
            })

    # Exits not visible — still flag blocked ones from known positions
    visible_coords = {(vx, vy) for vx, vy in visible}
    for ex in exit_positions:
        ex_id = f"exit_{ex[0]}_{ex[1]}"
        if (ex[0], ex[1]) not in visible_coords:
            if fire_grid[_idx(ex[0], ex[1], w)] >= EXIT_BLOCKED_FIRE_THRESHOLD:
                if ex_id not in blocked_exit_ids:
                    blocked_exit_ids.append(ex_id)

    # --- Audible signals ---
    audible: List[str] = []
    any_fire = any(fire_grid[i] >= FIRE_BURNING for i in range(w * h))
    if any_fire:
        audible.append("Fire alarm sounding")
    if smoke_label in ("moderate", "heavy"):
        audible.append("Smoke detector beeping")
    if agent_health < 50:
        audible.append("Your own laboured breathing")

    # --- Zone label ---
    location_label = zone_map.get(f"{agent_x},{agent_y}", "unknown area")

    # --- Action hints ---
    action_hints = _build_action_hints(
        agent_x, agent_y, cell_grid, visible,
        visible_objects, door_registry, w, h
    )

    # --- Narrative ---
    narrative = _compose_narrative(
        location_label=location_label,
        smoke_label=smoke_label,
        fire_visible=fire_visible,
        fire_dir=fire_dir,
        agent_health=agent_health,
        health_label=health_label,
        wind_dir=wind_dir,
        visible_objects=visible_objects,
        blocked_exit_ids=blocked_exit_ids,
        audible=audible,
        last_action_feedback=last_action_feedback,
        action_hints=action_hints,
    )

    return {
        "narrative": narrative,
        "agent_evacuated": agent_evacuated,
        "location_label": location_label,
        "smoke_level": smoke_label,
        "fire_visible": fire_visible,
        "fire_direction": fire_dir,
        "agent_health": agent_health,
        "health_status": health_label,
        "wind_dir": wind_dir,
        "visible_objects": visible_objects,
        "blocked_exit_ids": blocked_exit_ids,
        "audible_signals": audible,
        "elapsed_steps": step_count,
        "last_action_feedback": last_action_feedback,
        "available_actions_hint": action_hints,
        "done": False,
        "reward": 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_pos_str(ax: int, ay: int, tx: int, ty: int) -> str:
    dx, dy = tx - ax, ty - ay
    dist = abs(dx) + abs(dy)
    if dist == 0:
        return "here"
    if abs(dx) >= abs(dy):
        return f"{dist}m {'east' if dx > 0 else 'west'}"
    else:
        return f"{dist}m {'south' if dy > 0 else 'north'}"


def _build_action_hints(
    ax: int, ay: int,
    cell_grid: List[int],
    visible: Set[Tuple[int, int]],
    visible_objects: List[Dict],
    door_registry: Dict[str, List[int]],
    w: int, h: int,
) -> List[str]:
    hints: List[str] = []

    # Movement
    for dx, dy, dirname in _CARDINAL:
        nx, ny = ax + dx, ay + dy
        if _in_bounds(nx, ny, w, h):
            ct = cell_grid[_idx(nx, ny, w)]
            if ct in (FLOOR, DOOR_OPEN, EXIT):
                hints.append(f"move(direction='{dirname}')")

    # Door actions
    for obj in visible_objects:
        if obj["type"] == "door":
            did = obj["id"]
            if "closed" in obj["state"]:
                hints.append(f"door(target_id='{did}', door_state='open')")
            else:
                hints.append(f"door(target_id='{did}', door_state='close')")

    hints.append("wait()")
    return hints


def _compose_narrative(
    location_label: str,
    smoke_label: str,
    fire_visible: bool,
    fire_dir: Optional[str],
    agent_health: float,
    health_label: str,
    wind_dir: str,
    visible_objects: List[Dict],
    blocked_exit_ids: List[str],
    audible: List[str],
    last_action_feedback: str,
    action_hints: List[str],
) -> str:
    lines = []

    # Location + atmosphere
    lines.append(f"You are in the **{location_label}**. The air is **{smoke_label}**.")

    # Health + wind
    health_bar = _health_bar(agent_health)
    wind_str = f"Wind: **{wind_dir}**" if wind_dir != "CALM" else "Wind: calm"
    lines.append(f"Health: {health_bar} ({health_label})  |  {wind_str}")

    # Fire
    if fire_visible and fire_dir:
        lines.append(f"Flames are visible to the **{fire_dir}**.")
    else:
        lines.append("No fire directly visible.")

    # Objects (exits and doors)
    exits_vis = [o for o in visible_objects if o["type"] == "exit"]
    doors_vis = [o for o in visible_objects if o["type"] == "door"]

    if exits_vis:
        exit_descs = []
        for o in exits_vis:
            status = " **[BLOCKED]**" if o["state"].startswith("BLOCKED") else ""
            exit_descs.append(f"{o['id']}{status} at {o['relative_pos']}")
        lines.append(f"Exit{'s' if len(exits_vis) > 1 else ''} visible: {', '.join(exit_descs)}.")

    if doors_vis:
        door_descs = [f"{o['id']} ({o['state']}) at {o['relative_pos']}" for o in doors_vis]
        lines.append(f"Door{'s' if len(doors_vis) > 1 else ''}: {', '.join(door_descs)}.")

    if blocked_exit_ids:
        lines.append(f"WARNING: {len(blocked_exit_ids)} exit(s) blocked by fire — find an alternative route.")

    # Sound
    if audible:
        lines.append(f"You hear: {'; '.join(audible)}.")

    # Last action
    if last_action_feedback:
        lines.append(f"Last action: {last_action_feedback}")

    # Available actions
    if action_hints:
        hints_str = "  ".join(action_hints[:8])
        lines.append(f"Available actions: {hints_str}")

    return "\n".join(lines)


def _health_bar(health: float) -> str:
    filled = int(health / 10)
    empty = 10 - filled
    return "█" * filled + "░" * empty + f" {int(health)}/100"


def _terminal_obs(
    step_count: int,
    last_action_feedback: str,
    narrative: str,
    agent_evacuated: bool = False,
) -> Dict[str, Any]:
    return {
        "narrative": narrative,
        "agent_evacuated": agent_evacuated,
        "location_label": "",
        "smoke_level": "none",
        "fire_visible": False,
        "fire_direction": None,
        "agent_health": 0.0 if not agent_evacuated else 100.0,
        "health_status": "Critical" if not agent_evacuated else "Good",
        "wind_dir": "CALM",
        "visible_objects": [],
        "blocked_exit_ids": [],
        "audible_signals": [],
        "elapsed_steps": step_count,
        "last_action_feedback": last_action_feedback,
        "available_actions_hint": [],
        "done": True,
        "reward": 0.0,
    }
