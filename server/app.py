"""FastAPI application for the Pyre Environment.

Uses a factory function so each WebSocket session gets an isolated environment instance.

Configuration via environment variables:
  PYRE_MAX_STEPS   max steps before timeout (default 150)
  PYRE_SEED        base random seed (default 42)
  PORT             server port (default 8000)
"""

import os

from openenv.core.env_server.http_server import create_app

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


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    port = int(os.getenv("PORT", port))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
