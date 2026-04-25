"""Pyre Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import PyreAction, PyreObservation, PyreState


class PyreEnv(EnvClient[PyreAction, PyreObservation, PyreState]):
    """Client for the Pyre Environment.

    The environment is async by default; use .sync() for synchronous access:

        with PyreEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            print(result.observation.narrative)
            result = env.step(PyreAction(action="move", direction="north"))
            print(f"Health: {result.observation.agent_health}")

    Or use async:

        async with PyreEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
    """

    def _step_payload(self, action: PyreAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[PyreObservation]:
        obs_data = payload.get("observation", payload)
        obs = PyreObservation(
            narrative=obs_data.get("narrative", ""),
            agent_evacuated=obs_data.get("agent_evacuated", False),
            location_label=obs_data.get("location_label", ""),
            smoke_level=obs_data.get("smoke_level", "none"),
            fire_visible=obs_data.get("fire_visible", False),
            fire_direction=obs_data.get("fire_direction"),
            agent_health=obs_data.get("agent_health", 100.0),
            health_status=obs_data.get("health_status", "Good"),
            wind_dir=obs_data.get("wind_dir", "CALM"),
            visible_objects=obs_data.get("visible_objects", []),
            blocked_exit_ids=obs_data.get("blocked_exit_ids", []),
            audible_signals=obs_data.get("audible_signals", []),
            elapsed_steps=obs_data.get("elapsed_steps", 0),
            last_action_feedback=obs_data.get("last_action_feedback", ""),
            available_actions_hint=obs_data.get("available_actions_hint", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=payload.get("metadata", {}),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> PyreState:
        return PyreState(**payload)
