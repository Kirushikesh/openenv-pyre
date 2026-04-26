"""FastAPI application for the Pyre Environment.

Uses a factory function so each WebSocket session gets an isolated environment instance.

Configuration via environment variables:
  PYRE_MAX_STEPS   max steps before timeout (default 150)
  PYRE_SEED        base random seed (default 42)
  PORT             server port (default 8000)
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import Field, BaseModel
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from openenv.core.env_server.http_server import create_app
from starlette.routing import Route

try:
    from ..models import PyreAction, PyreObservation
    from .pyre_env_environment import PyreEnvironment
    from .narrative import build_narrative_observation, compute_visible_cells
except (ImportError, ModuleNotFoundError):
    from models import PyreAction, PyreObservation
    from server.pyre_env_environment import PyreEnvironment
    from server.narrative import build_narrative_observation, compute_visible_cells

MAX_STEPS = int(os.getenv("PYRE_MAX_STEPS", "150"))
BASE_SEED = int(os.getenv("PYRE_SEED", "42"))


def create_pyre_environment() -> PyreEnvironment:
    return PyreEnvironment(max_steps=MAX_STEPS, base_seed=BASE_SEED)


app = create_app(
    create_pyre_environment,
    PyreAction,
    PyreObservation,
    env_name="pyre_env",
)


_stateful_env: Optional[PyreEnvironment] = None


def get_stateful_env() -> PyreEnvironment:
    """Return singleton env used by HTTP reset/step for browser workflows."""
    global _stateful_env
    if _stateful_env is None:
        _stateful_env = create_pyre_environment()
    return _stateful_env


# Remove stateless HTTP routes from create_app so these stateful overrides are used.
app.routes[:] = [
    r
    for r in app.routes
    if not (
        isinstance(r, Route)
        and (
            (r.path in {"/reset", "/step"} and "POST" in (r.methods or set()))
            or (r.path == "/state" and "GET" in (r.methods or set()))
        )
    )
]


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: str = "medium"


class StepRequest(BaseModel):
    action: str = Field(..., description="Action type: move | door | wait | look")
    direction: Optional[str] = Field(None, description="Cardinal direction for move/look")
    target_id: Optional[str] = Field(None, description="Door ID for door action")
    door_state: Optional[str] = Field(None, description="'open' or 'close' for door action")


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def serve_frontend() -> FileResponse:
    """Serve the React frontend from server/static/index.html."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        # Fallback to the RPG viewer if index.html is missing
        rpg_path = STATIC_DIR / "viewer_rpg.html"
        if rpg_path.exists():
            return FileResponse(str(rpg_path))
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(str(html_path))


# Mount the static directory for assets (CSS, JS, etc.)
if (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


@app.post("/reset")
def reset_episode(body: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    env = get_stateful_env()
    obs = env.reset(seed=body.seed, difficulty=body.difficulty)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "metadata": obs.metadata or {},
    }


@app.post("/step")
def step_episode(body: StepRequest) -> Dict[str, Any]:
    env = get_stateful_env()
    if getattr(env, "_fire_sim", None) is None:
        raise HTTPException(status_code=409, detail="No active episode. Call POST /reset first.")
    obs = env.step(PyreAction(
        action=body.action,
        direction=body.direction,
        target_id=body.target_id,
        door_state=body.door_state,
    ))
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "metadata": obs.metadata or {},
    }


@app.get("/state")
def get_state() -> Dict[str, Any]:
    env = get_stateful_env()
    return env.state.model_dump()


@app.get("/scene")
def get_scene() -> Dict[str, Any]:
    """Return a compact scene snapshot for external frontends.

    Response shape
    --------------
    labels
        agent        — position, health, status flags, perception summary
        episode      — fire parameters, step counters, difficulty
        map          — grid dimensions, exit positions, door registry
        surroundings — visible objects, blocked exits, audible signals,
                       available action hints
    graph
        channels     — ordered list of channel names (index guide)
        channel_info — human-readable description of each channel
        width / height
        grid         — grid[y][x] = [cell_type, fire, smoke, is_agent, is_visible]
                       cell_type: 0=floor 1=wall 2=door_open 3=door_closed
                                  4=exit 5=obstacle
                       fire / smoke: 0.0 (none) → 1.0 (max)
                       is_agent / is_visible: 0 or 1
    """
    env = get_stateful_env()
    st = env.state

    # --- Build structured observation fields (no narrative) ---
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
        last_action_feedback=getattr(env, "_last_feedback", ""),
        wind_dir=st.wind_dir,
        w=st.grid_w,
        h=st.grid_h,
    )

    # --- Visibility set for the graph layer ---
    if st.agent_alive and not st.agent_evacuated:
        visible_set = compute_visible_cells(
            st.agent_x, st.agent_y,
            st.cell_grid, st.smoke_grid,
            st.grid_w, st.grid_h,
        )
    else:
        visible_set = set()

    # --- Labels ---
    labels: Dict[str, Any] = {
        "agent": {
            "x": st.agent_x,
            "y": st.agent_y,
            "health": st.agent_health,
            "health_status": obs_data.get("health_status", "Good"),
            "alive": st.agent_alive,
            "evacuated": st.agent_evacuated,
            "location": obs_data.get("location_label", ""),
            "smoke_level": obs_data.get("smoke_level", "none"),
            "fire_visible": obs_data.get("fire_visible", False),
            "fire_direction": obs_data.get("fire_direction", None),
            "last_action_feedback": obs_data.get("last_action_feedback", ""),
        },
        "episode": {
            "id": st.episode_id,
            "step": st.step_count,
            "max_steps": st.max_steps,
            "template": st.template_name,
            "difficulty": getattr(env, "_difficulty", "medium"),
            "wind_dir": st.wind_dir,
            "fire_spread_rate": st.fire_spread_rate,
            "humidity": st.humidity,
            "fire_sources": st.fire_sources_count,
        },
        "map": {
            "width": st.grid_w,
            "height": st.grid_h,
            "exit_positions": st.exit_positions,
            "door_registry": st.door_registry,
        },
        "surroundings": {
            "visible_objects": obs_data.get("visible_objects", []),
            "blocked_exit_ids": obs_data.get("blocked_exit_ids", []),
            "audible_signals": obs_data.get("audible_signals", []),
            "available_actions": obs_data.get("available_actions_hint", []),
        },
    }

    # --- 2-D multi-channel grid ---
    w, h = st.grid_w, st.grid_h
    grid: List[List[List[float]]] = []
    for y in range(h):
        row: List[List[float]] = []
        for x in range(w):
            idx = y * w + x
            cell_type = float(st.cell_grid[idx])
            fire      = round(st.fire_grid[idx], 4)
            smoke     = round(st.smoke_grid[idx], 4)
            is_agent  = 1.0 if (x == st.agent_x and y == st.agent_y) else 0.0
            is_visible = 1.0 if (x, y) in visible_set else 0.0
            row.append([cell_type, fire, smoke, is_agent, is_visible])
        grid.append(row)

    graph: Dict[str, Any] = {
        "channels": ["cell_type", "fire", "smoke", "is_agent", "is_visible"],
        "channel_info": {
            "cell_type": "0=floor 1=wall 2=door_open 3=door_closed 4=exit 5=obstacle",
            "fire":      "0.0=none to 1.0=fully burning",
            "smoke":     "0.0=clear to 1.0=dense smoke",
            "is_agent":  "1 if agent occupies this cell, else 0",
            "is_visible": "1 if within agent line-of-sight, else 0",
        },
        "width":  w,
        "height": h,
        "grid":   grid,
    }

    return {"labels": labels, "graph": graph}


async def event_generator(request: Request) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator for Server-Sent Events."""
    while True:
        if await request.is_disconnected():
            break
        
        try:
            # Re-use the logic from get_scene() but in an async-friendly way
            # We call get_scene directly here as it's a simple synchronous function
            # and won't block the event loop for long.
            scene_data = get_scene()
            yield {"data": json.dumps(scene_data)}
        except Exception as e:
            yield {"data": json.dumps({"error": str(e)})}
            
        await asyncio.sleep(0.5)  # Update every 500ms as requested


@app.get("/live-movements")
async def stream_movements(request: Request):
    return EventSourceResponse(event_generator(request))


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    port = int(os.getenv("PORT", port))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
