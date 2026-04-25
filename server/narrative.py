"""
Observation rendering for Pyre.

Converts raw server state into:
  - A first-person narrative string (the LLM's primary input)
  - Structured fields in PyreObservation

Visibility rules:
  - Base radius: 5 (Manhattan distance)
  - Moderate smoke in agent's cell: radius 3
  - Heavy smoke in agent's cell: radius 2
  - Walls block flood-fill propagation
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .fire_sim import smoke_level_label, FIRE_BURNING

# Cell type constants
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


# ---------------------------------------------------------------------------
# Visibility computation
# ---------------------------------------------------------------------------

def compute_visible_cells(
    ax: int,
    ay: int,
    cell_grid: List[int],
    smoke_grid: List[float],
    w: int,
    h: int,
) -> Set[Tuple[int, int]]:
    """BFS flood-fill from agent position; walls block propagation.

    Returns set of (x, y) the agent can currently perceive.
    """
    agent_smoke = smoke_grid[_idx(ax, ay, w)]
    smoke_label = smoke_level_label(agent_smoke)

    if smoke_label == "heavy":
        radius = 2
    elif smoke_label == "moderate":
        radius = 3
    else:
        radius = 5

    visible: Set[Tuple[int, int]] = {(ax, ay)}
    queue = [(ax, ay, 0)]  # (x, y, dist)

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
                continue  # walls block LOS and are themselves not visible
            visible.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return visible


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def build_narrative_observation(
    step_count: int,
    agent_x: int,
    agent_y: int,
    agent_alive: bool,
    agent_evacuated: bool,
    cell_grid: List[int],
    fire_grid: List[float],
    smoke_grid: List[float],
    npcs: List[Dict[str, Any]],
    exit_positions: List[List[int]],
    door_registry: Dict[str, List[int]],
    zone_map: Dict[str, str],
    last_action_feedback: str,
    w: int,
    h: int,
) -> Dict[str, Any]:
    """Build the full observation dict (matches PyreObservation fields).

    Returns a dict suitable for PyreObservation(**result).
    """
    if agent_evacuated:
        return _terminal_obs(step_count, last_action_feedback, done=True, reward=0.0,
                             narrative="You have safely evacuated the building.")

    if not agent_alive:
        return _terminal_obs(step_count, last_action_feedback, done=True, reward=0.0,
                             narrative="You have been overcome by fire and smoke.")

    visible = compute_visible_cells(agent_x, agent_y, cell_grid, smoke_grid, w, h)

    # --- Smoke level at agent cell ---
    agent_smoke = smoke_grid[_idx(agent_x, agent_y, w)]
    smoke_label = smoke_level_label(agent_smoke)

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
                # Determine cardinal direction of nearest fire
                dx = vx - agent_x
                dy = vy - agent_y
                if abs(dx) >= abs(dy):
                    fire_dir = "east" if dx > 0 else "west"
                else:
                    fire_dir = "south" if dy > 0 else "north"

    # --- Visible NPCs ---
    npc_set = {(int(n["x"]), int(n["y"])): n for n in npcs}
    visible_people: List[Dict[str, Any]] = []
    for vx, vy in visible:
        if (vx, vy) == (agent_x, agent_y):
            continue
        npc = npc_set.get((vx, vy))
        if npc:
            rel = _relative_pos_str(agent_x, agent_y, vx, vy)
            visible_people.append({
                "id": npc["id"],
                "relative_pos": rel,
                "state": npc["state"],
                "last_seen_step": step_count,
            })

    # --- Visible objects (doors and exits) ---
    visible_objects: List[Dict[str, Any]] = []
    door_pos_to_id = {(v[0], v[1]): k for k, v in door_registry.items()}
    for vx, vy in visible:
        ct = cell_grid[_idx(vx, vy, w)]
        rel = _relative_pos_str(agent_x, agent_y, vx, vy)
        if ct in (DOOR_OPEN, DOOR_CLOSED):
            door_id = door_pos_to_id.get((vx, vy), f"door_{vx}_{vy}")
            door_state = "open" if ct == DOOR_OPEN else "closed"
            # Annotate hot doors
            if fire_grid[_idx(vx, vy, w)] > 0.1:
                door_state += " (hot)"
            visible_objects.append({
                "id": door_id,
                "type": "door",
                "relative_pos": rel,
                "state": door_state,
            })
        elif ct == EXIT:
            visible_objects.append({
                "id": f"exit_{vx}_{vy}",
                "type": "exit",
                "relative_pos": rel,
                "state": "open",
            })

    # --- Audible signals ---
    audible: List[str] = []
    # Alarm if any fire is burning in the map
    any_fire = any(fire_grid[i] >= FIRE_BURNING for i in range(w * h))
    if any_fire:
        audible.append("Fire alarm sounding")
    # Screams from panicked NPCs within extended radius (7)
    panicked_nearby = [
        n for n in npcs
        if n["state"] in ("panicked", "injured")
        and _manhattan(agent_x, agent_y, int(n["x"]), int(n["y"])) <= 7
    ]
    if panicked_nearby:
        audible.append(f"Screaming from nearby")

    # --- Zone label ---
    location_label = zone_map.get(f"{agent_x},{agent_y}", "unknown area")

    # --- Action hints ---
    action_hints = _build_action_hints(
        agent_x, agent_y, cell_grid, visible,
        visible_people, visible_objects, door_registry, exit_positions, w, h
    )

    # --- Narrative string ---
    narrative = _compose_narrative(
        location_label=location_label,
        smoke_label=smoke_label,
        fire_visible=fire_visible,
        fire_dir=fire_dir,
        visible_people=visible_people,
        visible_objects=visible_objects,
        audible=audible,
        last_action_feedback=last_action_feedback,
        action_hints=action_hints,
    )

    return {
        "narrative": narrative,
        "location_label": location_label,
        "smoke_level": smoke_label,
        "fire_visible": fire_visible,
        "fire_direction": fire_dir,
        "visible_people": visible_people,
        "visible_objects": visible_objects,
        "audible_signals": audible,
        "elapsed_steps": step_count,
        "last_action_feedback": last_action_feedback,
        "available_actions_hint": action_hints,
        "done": False,
        "reward": 0.0,  # filled by environment
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_pos_str(ax: int, ay: int, tx: int, ty: int) -> str:
    dx, dy = tx - ax, ty - ay
    dist = abs(dx) + abs(dy)
    if abs(dx) >= abs(dy):
        horiz = f"{abs(dx)}m {'east' if dx > 0 else 'west'}"
        return f"{horiz}" if dist <= 3 else f"{dist}m {'east' if dx > 0 else 'west'}"
    else:
        vert = f"{abs(dy)}m {'south' if dy > 0 else 'north'}"
        return f"{vert}" if dist <= 3 else f"{dist}m {'south' if dy > 0 else 'north'}"


def _build_action_hints(
    ax: int,
    ay: int,
    cell_grid: List[int],
    visible: Set[Tuple[int, int]],
    visible_people: List[Dict],
    visible_objects: List[Dict],
    door_registry: Dict[str, List[int]],
    exit_positions: List[List[int]],
    w: int,
    h: int,
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
                hints.append(f"open_door(target_id='{did}')")
            else:
                hints.append(f"close_door(target_id='{did}')")

    # Instruct NPCs
    for person in visible_people[:3]:  # cap hints
        pid = person["id"]
        hints.append(f"instruct(target_id='{pid}', direction='south')")

    # Broadcast (always available if there are active NPCs nearby)
    if visible_people:
        hints.append("broadcast(zone='main_corridor', category='evacuate_south')")

    hints.append("wait()")
    return hints


def _compose_narrative(
    location_label: str,
    smoke_label: str,
    fire_visible: bool,
    fire_dir: Optional[str],
    visible_people: List[Dict],
    visible_objects: List[Dict],
    audible: List[str],
    last_action_feedback: str,
    action_hints: List[str],
) -> str:
    lines = []

    # Location + atmosphere
    lines.append(f"You are in the **{location_label}**. The air is **{smoke_label}**.")

    # Fire
    if fire_visible and fire_dir:
        lines.append(f"Flames are visible to the **{fire_dir}**.")
    else:
        lines.append("No fire directly visible.")

    # People
    if visible_people:
        descs = []
        for p in visible_people:
            descs.append(f"{p['id']} ({p['state']}) is {p['relative_pos']}")
        lines.append(f"{len(visible_people)} {'person' if len(visible_people) == 1 else 'people'} nearby: {', '.join(descs)}.")
    else:
        lines.append("No one visible nearby.")

    # Objects
    exits = [o for o in visible_objects if o["type"] == "exit"]
    doors = [o for o in visible_objects if o["type"] == "door"]
    if exits:
        exit_descs = [f"exit at {o['relative_pos']}" for o in exits]
        lines.append(f"Exit{'s' if len(exits) > 1 else ''} visible: {', '.join(exit_descs)}.")
    if doors:
        door_descs = [f"{o['id']} ({o['state']}) at {o['relative_pos']}" for o in doors]
        lines.append(f"Door{'s' if len(doors) > 1 else ''}: {', '.join(door_descs)}.")

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


def _terminal_obs(
    step_count: int,
    last_action_feedback: str,
    done: bool,
    reward: float,
    narrative: str,
) -> Dict[str, Any]:
    return {
        "narrative": narrative,
        "location_label": "",
        "smoke_level": "none",
        "fire_visible": False,
        "fire_direction": None,
        "visible_people": [],
        "visible_objects": [],
        "audible_signals": [],
        "elapsed_steps": step_count,
        "last_action_feedback": last_action_feedback,
        "available_actions_hint": [],
        "done": done,
        "reward": reward,
    }
