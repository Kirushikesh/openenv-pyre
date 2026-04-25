# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Start the server (http://localhost:8000)
uv run server

# Run the random baseline agent (smoke test, 5 episodes)
python examples/random_agent.py --episodes 5 --verbose

# Health check
curl http://localhost:8000/health

# Reset episode
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"difficulty": "medium"}'

# Execute action
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"action": "move", "direction": "north"}'
```

Environment variables: `PYRE_MAX_STEPS` (default 150), `PYRE_SEED` (default 42), `PORT` (default 8000).

## Architecture

**Pyre** is an RL training environment where an LLM agent navigates a burning building with partial observability. The agent receives only first-person text observations — no global map — and must reach exits before dying or timing out.

### Simulation Layer
- **[`server/fire_sim.py`](server/fire_sim.py)** — Cellular automaton (16×16 grid). Per-step: compute ignitions (stochastic, wind-biased), advance fire intensity, spread smoke. Closed doors reduce fire spread to 15% (`DOOR_CLOSED_FIRE_FACTOR = 0.15`). Burnout converts cells to `OBSTACLE`. Difficulty controls `p_spread`, `wind_dir`, `humidity`, `max_steps`.
- **[`server/floor_plan.py`](server/floor_plan.py)** — Three hand-authored 16×16 building templates: `small_office` (two-corridor), `open_plan` (open hall with pillars), `t_corridor` (T-shaped hallways). Each template carries `fuel_map` (offices burn hotter at 1.5×), `ventilation_map` (open areas clear smoke faster at 0.050 vs. 0.010 in rooms), `zone_map`, and `agent_spawn_options`. `generate_episode()` randomizes door states, NPC positions, and fire start locations.

### Observation Layer
- **[`server/narrative.py`](server/narrative.py)** — Converts server state to first-person text. Visibility is a BFS flood-fill blocked by walls; radius shrinks under smoke (5 → 3 → 2). Reports smoke level, fire direction, visible doors/exits (blocked if fire ≥ 0.5), audible signals, and action hints as structured text + structured fields.

### Reward Layer
- **[`server/rubrics.py`](server/rubrics.py)** — Eight composable `RubricComponent` classes each with a `.score()` method. Per-step: `TimeStepPenalty` (−0.01), `ProgressReward` (+0.1, BFS-based distance to exit), `DangerPenalty` (−0.5), `HealthDrainPenalty` (−0.02×damage), `StrategicDoorBonus` (+0.5). Episode-end: `SelfSurviveBonus` (+5.0), `SelfDeathPenalty` (−10.0), `TimeBonus` (+0.05×remaining_steps).

### Orchestration
- **[`server/pyre_env_environment.py`](server/pyre_env_environment.py)** — `PyreEnvironment` is the main state machine. `reset()` builds the episode; `step()` executes action → advances fire → damages health → computes reward via rubrics → builds observation via `narrative.py`. Health damage: smoke (0.5–5 HP/step), fire (10 HP/step). Done when: evacuated, health ≤ 0, or steps ≥ max_steps. Difficulty presets are in `_DIFFICULTY_PRESETS`.
- **[`models.py`](models.py)** — Pydantic data contracts: `PyreAction` (move/door/look/wait), `PyreObservation` (what agents see), `PyreMapState` (full grid snapshot for UI), `PyreState` (server ground truth, not exposed to agents).

### Interface
- **[`server/app.py`](server/app.py)** — FastAPI bootstrap via OpenEnv `create_app` factory. Entry point: `pyre_env.server.app:main`.
- **[`client.py`](client.py)** — `PyreEnv(EnvClient)` async/sync Python client. Parses responses into typed `PyreObservation` objects.
- **[`examples/random_agent.py`](examples/random_agent.py)** — Baseline agent; parses action hints from observations, 70% hint-biased / 30% random. Use for smoke-testing environment changes.

### Data Flow
```
reset()/step() → PyreEnvironment
  → fire_sim.step()           # advance physics
  → narrative.build_obs()     # text + structured obs
  → rubrics.score()           # reward signal
  → PyreObservation           # returned to agent
```

### Grid Conventions
- All grids are 16×16, stored as row-major flat lists (index = `y * 16 + x`)
- Cell types: `EMPTY`, `WALL`, `OBSTACLE`, `DOOR_OPEN`, `DOOR_CLOSED`, `EXIT`
- Fire/smoke values are floats in `[0, 1]`; fire ≥ 0.5 blocks exits
