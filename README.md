---
title: Pyre — Crisis Navigation Environment
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Pyre — Crisis Navigation Environment for LLM Agents

> *When buildings burn, the difference between a safe evacuation and a tragedy is the quality of decisions made in the first 60 seconds. Can we train an LLM to be the right kind of guide?*

**Pyre** places an LLM agent *inside* a burning building. The agent must navigate to safety while simultaneously coordinating nearby civilians toward exits, managing fire spread through door control, and reasoning under partial observability — all with no global map and hard time pressure.

---

## Why Pyre vs. existing environments

| Feature | `grid_world` | `maze_env` | `wildfire_env` | **Pyre** |
|---|---|---|---|---|
| Observability | Full | Full | Partial | **Partial, first-person, text** |
| Map dynamics | Static | Static | Dynamic (fire) | **Dynamic (fire + doors + crowd)** |
| Other entities | None | None | Few | **Many NPCs with behavior model** |
| Action richness | 4 moves | 4 moves | Suppression | **Movement + door control + speech acts** |
| Agent role | Mover | Mover | Suppressor | **Coordinator + survivor** |
| Reward complexity | Reach goal | Reach goal | Suppress fire | **13-component composite rubric** |

*`wildfire_env` trains an agent to fight fires from above; Pyre trains an agent to survive and lead others out from inside.*

---

## What the agent sees (narrative observation)

Every step the agent receives a first-person text observation:

```
You are in the **main_corridor**. The air is **moderate**.
Flames are visible to the **east**.
3 people nearby: p_3 (panicked) is 2m north, p_7 (calm) is 3m west, p_1 (injured) is 1m east.
Exit visible: exit at 8m west.
Doors: door_2 (closed) at 2m east.
You hear: Fire alarm sounding; Screaming from nearby.
Last action: You move south. The smoke is thick here.
Available actions: move(direction='north')  move(direction='west')  close_door(target_id='door_2')  instruct(target_id='p_3', direction='west')  wait()
```

---

## Action space

| Action | Parameters | Effect |
|---|---|---|
| `move` | `direction` | Move one cell N/S/E/W |
| `instruct` | `target_id`, `direction` | Direct nearby NPC (compliance depends on NPC state) |
| `close_door` | `target_id` | Close door — slows fire 7× but may trap NPCs |
| `open_door` | `target_id` | Open a closed door |
| `broadcast` | `zone`, `category` | Instruct all NPCs in a zone at once |
| `wait` | — | Skip turn |

---

## Reward function (composite rubric)

**Per step:**
- `-0.01` constant time penalty
- `+0.1` moved closer to exit
- `-0.5` moved into heavy smoke or fire-adjacent cell
- `+0.2` issued instruction an NPC followed toward exit
- `-0.05` issued instruction no one followed
- `+0.5` closed door adjacent to active fire (strategic)
- `-2.0` closed door that later trapped a casualty NPC

**Episode end:**
- `+5.0` agent evacuated alive
- `-10.0` agent incapacitated
- `+1.0 × N` per NPC evacuated
- `-2.0 × N` per NPC casualty
- `+3.0` no stampede occurred
- `-1.5 × N` per stampede event

---

## Quick start

```bash
cd pyre_env
uv sync
uv run server   # → http://localhost:8000

# Health check
curl http://localhost:8000/health

# Reset
curl -X POST http://localhost:8000/reset

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
    print(f"Reward: {result.reward}")
```

---

## Deployment

```bash
openenv push --repo-id your-org/pyre-env
```

---

## Project structure

```
pyre_env/
├── models.py                       PyreAction, PyreObservation, PyreState
├── client.py                       PyreEnv (EnvClient subclass)
├── openenv.yaml                    OpenEnv manifest
├── pyproject.toml
├── server/
│   ├── app.py                      FastAPI bootstrap
│   ├── pyre_env_environment.py     Main Environment class
│   ├── floor_plan.py               3 building templates + episode generation
│   ├── fire_sim.py                 Cellular automaton fire/smoke
│   ├── npc_model.py                NPC state machine + stampede detection
│   ├── narrative.py                Visibility + text observation renderer
│   └── rubrics.py                  13 composable reward components
└── examples/
    ├── random_agent.py             Smoke-test baseline
    └── pyre_grpo_training.ipynb    GRPO training notebook (TRL + Unsloth)
```

---

## Hackathon alignment

- **Theme #2 — Long-Horizon Planning**: 50–150 step episodes; agent must build a mental map across many observations
- **Theme #3.1 — World Modeling**: no global map; agent infers fire spread, NPC locations, and corridor topology from local text observations
