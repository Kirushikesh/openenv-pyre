---
title: Pyre — Crisis Navigation Environment
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Pyre — Crisis Navigation Environment for LLM Agents

> *When buildings burn, the difference between a safe evacuation and a tragedy is the quality of decisions made in the first 60 seconds. Can we train an LLM to make them?*

**Pyre** places an LLM agent *inside* a burning building. The agent must navigate to safety under partial observability — no global map, hard time pressure, and a fire that actively spreads and blocks exits.

---

## Why Pyre vs. existing environments

| Feature | `grid_world` | `maze_env` | `wildfire_env` | **Pyre** |
|---|---|---|---|---|
| Observability | Full | Full | Partial | **Partial, first-person, text** |
| Map dynamics | Static | Static | Dynamic (fire) | **Dynamic (fire + doors)** |
| Action richness | 4 moves | 4 moves | Suppression | **Movement + door control + look** |
| Agent role | Mover | Mover | Suppressor | **Survivor** |
| Reward complexity | Reach goal | Reach goal | Suppress fire | **8-component composite rubric** |

*`wildfire_env` trains an agent to fight fires from above; Pyre trains an agent to survive from inside.*

---

## What the agent sees (narrative observation)

Every step the agent receives a first-person text observation:

```
You are in the **main_corridor**. The air is **moderate**.
Health: ████████░░ (85/100) | Wind: **EAST**
Flames are visible to the **west**.
Exits visible: exit_0_7 at 8m west.
Doors: door_1 (closed) at 2m east.
You hear: Fire alarm sounding; Smoke detector beeping.
Last action: You move south. The smoke is thick here.
Available actions: move(direction='north')  move(direction='south')  door(target_id='door_1', door_state='open')  look(direction='east')  wait()
```

---

## Action space

| Action | Parameters | Effect |
|---|---|---|
| `move` | `direction` | Move one cell N/S/E/W |
| `door` | `target_id`, `door_state` | Open or close a nearby door — closed doors slow fire spread to 15% |
| `look` | `direction` | Scan up to 5 cells in one direction for detailed zone/fire/smoke info |
| `wait` | — | Skip turn |

---

## Reward function (composite rubric)

**Per step:**
- `-0.01` constant time penalty
- `+0.1` moved closer to nearest unblocked exit (BFS distance)
- `-0.5` moved into smoke ≥ moderate or fire-adjacent cell
- `-0.02 × damage` health drain from smoke/fire exposure
- `+0.5` closed a door adjacent to active fire (strategic)

**Episode end:**
- `+5.0` agent evacuated alive
- `-10.0` agent incapacitated
- `+0.05 × remaining_steps` time bonus for fast evacuation

---

## Quick start

```bash
cd pyre_env
uv sync
uv run server   # → http://localhost:8000

# Health check
curl http://localhost:8000/health

# Reset (difficulty: easy | medium | hard)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium"}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "move", "direction": "north"}'

# Random baseline (5 episodes)
python examples/random_agent.py --episodes 5 --verbose
```

### Python client

```python
from pyre_env import PyreEnv, PyreAction

with PyreEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.narrative)
    result = env.step(PyreAction(action="move", direction="north"))
    print(f"Reward: {result.reward}, Health: {result.observation.agent_health}")
```

---

## Difficulty levels

| Level | Fire sources | Spread rate | Wind | Humidity | Max steps |
|---|---|---|---|---|---|
| `easy` | 1 | 10–20% | Calm | 30–50% | 200 |
| `medium` | 2–4 | 15–40% | Any | 10–45% | 150 |
| `hard` | 3–5 | 30–55% | Never calm | 5–20% | 100 |

---

## Deployment

```bash
openenv push --repo-id your-org/pyre-env
```

---

## Project structure

```
pyre_env/
├── models.py                       PyreAction, PyreObservation, PyreMapState, PyreState
├── client.py                       PyreEnv (EnvClient subclass)
├── openenv.yaml                    OpenEnv manifest
├── pyproject.toml
├── server/
│   ├── app.py                      FastAPI bootstrap
│   ├── pyre_env_environment.py     Main Environment class
│   ├── floor_plan.py               3 building templates + episode generation
│   ├── fire_sim.py                 Cellular automaton fire/smoke simulation
│   ├── narrative.py                Visibility + first-person text observation renderer
│   └── rubrics.py                  8 composable reward components
└── examples/
    └── random_agent.py             Smoke-test baseline
```

---

## Hackathon alignment

- **Theme #2 — Long-Horizon Planning**: 50–200 step episodes; agent must build a mental map across many partial observations
- **Theme #3.1 — World Modeling**: no global map; agent infers fire spread, corridor topology, and exit reachability from local text observations alone
