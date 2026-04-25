"""FastAPI application for the Pyre Environment.

Uses a factory function so each WebSocket session gets an isolated environment instance.

Configuration via environment variables:
  PYRE_MAX_STEPS   max steps before timeout (default 150)
  PYRE_SEED        base random seed (default 42)
  PORT             server port (default 8000)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import Field
from fastapi import HTTPException
from fastapi.responses import FileResponse
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel
from starlette.routing import Route

from ..models import PyreAction, PyreObservation
from .pyre_env_environment import PyreEnvironment

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
    """Serve the bundled RPG viewer from server/static/viewer_rpg.html."""
    html_path = STATIC_DIR / "viewer_rpg.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found: server/static/viewer_rpg.html")
    return FileResponse(str(html_path))


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


def main(host: str = "0.0.0.0", port: int = 8001):
    import uvicorn
    port = int(os.getenv("PORT", port))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
