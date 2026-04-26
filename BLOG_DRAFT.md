# Pyre: Teaching an LLM to Escape a Burning Building

*How we built a fire-evacuation environment with real physics, 14 reward components, and trained a language model to survive under partial observability.*

---

There's a specific kind of terror in a building fire that has nothing to do with the flames themselves. It's the smoke that fills a corridor in 40 seconds and halves your visibility. It's the exit you memorized that now has fire sitting on it. It's the door you left open because it seemed fine, and now the fire has traveled twenty feet because of it.

We wanted to know if an LLM could reason about that.

Not "here is a top-down map, find the shortest path" — that's a graph problem. We wanted the model to receive only what a panicking person actually has: a first-person description of the room they're standing in, the smoke level in their lungs, the direction fire was visible two seconds ago, and a list of doors they can reach. Then we wanted it to decide.

That became **Pyre**.

---

## Why text, why partial, why hard

The most natural interface for a language model is language. But the design choice to use text observations wasn't just convenience — it was the whole point.

Every step, the agent receives something like this:

```
You are in the **main_corridor**. The air is **moderate**.
Health: ████████░░ (85/100) | Wind: EAST
Flames are visible to the **west**.
Exits visible: exit_0_6 at 6m west.
Doors: door_2 (closed) at 2m east.
WARNING: exit_15_8 blocked by fire — find an alternative route.
You hear: Fire alarm sounding; Smoke detector beeping.
Last action: You move north. The smoke is thick here.
Available actions: move(direction='north')  move(direction='south')  door(target_id='door_2', door_state='open')  look(direction='east')  wait()
```

No coordinates. No grid. No god's-eye view of the fire spread. The agent has to build its own mental model of the building from successive partial glimpses — exactly what a world-modeling-capable LLM should be doing.

Visibility is computed as a BFS flood-fill from the agent's position, blocked by walls. The radius starts at 5 cells but shrinks dynamically: heavy smoke in your cell cuts it to 2. You can `look` in a direction to scan up to 5 cells ahead, but that costs a step — and every step the fire is spreading.

---

## The fire is a real simulation

This was non-negotiable for us. A scripted fire that moves on rails is trivial to learn; the agent just memorizes the path. We needed a fire that was genuinely unpredictable — where the agent had to reason about *where it might go*, not just *where it is*.

The fire simulation is a cellular automaton running on a 16×16 grid. Every step:

1. Any burning cell can ignite adjacent floor cells with probability `p_spread × (1 - humidity)`.
2. The wind biases that probability directionally — downwind spread gets a **2× multiplier**, upwind gets **0.5×**. A fire burning in an eastward wind races east and resists spreading west.
3. Office rooms have a fuel multiplier of **1.5×** — paper and wooden furniture burn faster than open corridors.
4. Open-plan halls have ventilation rates **5× higher** than closed rooms — smoke clears faster in the atrium, builds up in the offices.
5. A closed door reduces fire spread to **15% of normal**. This is not a game mechanic — it reflects the real physics of compartmentalization. Closing a door actually matters.
6. Burning cells accumulate a timer. After enough ticks they burn out and become impassable rubble.

What this produces is a fire that has character. An eastward wind with three ignition sources in the north offices will behave completely differently from a calm fire starting in the open-plan hall. The agent can't memorize trajectories — it has to understand cause and effect.

---

## Reward design is the hardest part

Getting the reward right took longer than the fire simulation. We ended up with **14 rubric components**, split across per-step and episode-end signals. Here's the reasoning behind each group.

### Per-step rewards

**Progress and anti-stagnation.** `ProgressReward` (+0.1) fires when the agent moves strictly closer to the nearest unblocked exit — measured by BFS traversal distance, not Manhattan. A shorter corridor that's now blocked by fire doesn't count. The symmetric `ProgressRegressionPenalty` (−0.05) discourages the agent from wandering backward. `SafeProgressBonus` (+0.05) stacks on top when that progress happens through a smoke-free cell — teaching the agent to prefer clean routes when multiple paths lead to the same exit.

`ExplorationBonus` (+0.02) gives a tiny reward for visiting a new cell. Without it, an agent that has been penalized for walking into smoke tends to freeze in the last "safe" spot it found. The exploration signal breaks that paralysis.

**Danger signals.** `DangerPenalty` (−0.5) fires when the agent moves into a cell with moderate/heavy smoke or adjacent to active fire. `HealthDrainPenalty` (−0.02 × damage) is proportional to actual HP lost — smoke costs about −0.04/step, being on fire costs −0.20/step. These two together create a continuous signal that encodes "this path is hurting you."

**The strategic door bonus.** `StrategicDoorBonus` (+0.5) rewards the agent for closing a door that is adjacent to active fire. The implementation has one critical guard: each door can only earn this bonus **once per episode**. Without that guard, the agent quickly discovers it can farm reward indefinitely by opening and re-closing the same door. With it, the only way to earn the bonus is to actually close a door that's protecting something — the model has to understand the *reason* to get the signal.

### Episode-end rewards

`SelfSurviveBonus` (+5.0) is the primary success signal. `HealthSurvivalBonus` (+1.5 × HP/100) rewards evacuating in better condition — an agent that escapes at 95 HP earns significantly more than one that barely makes it at 5 HP, which teaches the agent to prefer safer routes over shortest routes.

`SelfDeathPenalty` (−10.0) and `TimeoutPenalty` (−5.0) maintain the ordering: success > timeout > death. This is intentional — timing out while still alive is a failure, but it's a less catastrophic failure than dying. The gradient needs to reflect that.

The hardest reward engineering problem was **hard difficulty**. With faster fire spread, wind, and fewer max steps, early training produced a completely flat signal — every episode ended in death, every episode got −10.0, no gradient. We added `NearMissBonus`, which gives partial credit on failure based on the closest the agent got to an exit during the episode:

```
max(0.0, 3.0 - 0.5 × min_exit_dist_reached)
```

A run where the agent reached within 1 cell of the exit before dying earns +2.5. A run where the agent never got within 6 cells earns nothing. This converts the otherwise flat death landscape into a gradient that points the model toward "get close to exits" — which is the correct first thing to learn.

---

## Three buildings, procedurally more

We hand-authored three 16×16 floor plans to ensure the agent generalizes rather than memorizes:

- **small_office**: two horizontal corridors with rooms north and south, exits on opposite walls. Classic office building — the agent has to decide which corridor to take and whether the doors to the rooms are worth closing.
- **open_plan**: a large hall with pillar obstacles at the corners and diagonal exits. No internal doors — but the fire spreads faster because there's nothing to compartmentalize it.
- **t_corridor**: a T-shaped layout with a vertical stem and a horizontal bar, three exits. The agent may start deep in the stem and have to reason about whether the top exit or the side exits are less blocked.

Episode generation randomizes fire ignition count (1 on easy, 2–4 on medium, 3–5 on hard), spread rate, wind direction, humidity, agent spawn position, and door initial states. Two runs on the same floor plan will have entirely different fire dynamics.

We also built a procedural floor plan generator using a room-and-corridor algorithm (random rectangles + Prim-style MST corridors + exit tunneling) with a BFS connectivity guard — every generated map is guaranteed to have at least one reachable exit from every spawn point. The hard difficulty curriculum uses these procedural maps.

---

## Training: GRPO with a live environment server

We trained using **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL, with LoRA via Unsloth for efficiency.

The key design decision in the training loop is that **the reward signal is never leaked into the prompt**. The model receives only the narrative observation — it has to learn from the GRPO reward functions, not from being told "that was a good move." This keeps the gradient signal honest.

The output format is a two-part response: a `<think>...</think>` block for internal reasoning, followed by exactly one JSON action. The parser assigns a format score based on quality:

- 1.0 — valid JSON + `<think>` tags present
- 0.7 — valid JSON, no reasoning tags
- 0.4 — partial JSON rescued by regex
- 0.0 — completely unparseable → fallback to `wait`

This format score feeds into the reward signal, teaching the model not just what actions to take but how to structure its reasoning.

The training curriculum starts on `easy` difficulty only — one fire source, slow spread, calm wind, 200 max steps. The goal is to get the model to first understand the basic loop: observe, reason, move toward exits. Only after that does it make sense to introduce the harder dynamics.

---

## Results

*[Add actual GRPO training plots here — reward curve over training steps, evacuation rate over training steps, before/after episode comparison showing baseline vs. trained agent behavior. Screenshots of the trained agent's <think> reasoning blocks showing learned strategy are especially compelling.]*

---

## What we learned building this

**Reward collapse on hard mode is a real problem, not a theory.** We hit it. Every episode dying with −10.0 and no gradient. The `NearMissBonus` was added specifically to fix this and it made an immediate difference.

**The partial observability design is doing real work.** An agent with full map access would solve this as a shortest-path problem. With BFS-limited vision and smoke-shrinking radius, the agent has to commit to directions under uncertainty, use `look` actions strategically, and update its mental model as it moves. That's a genuinely harder cognitive task, and GRPO is the right tool for it because you can't easily write down the "correct" observation-action traces for every situation.

**Doors as strategic tools, not obstacles.** The `StrategicDoorBonus` was the reward component we were most unsure about. In practice, it's the component we're most proud of — it creates a non-obvious learned behavior (closing fire-adjacent doors to protect the path) that a random agent never discovers and that can't be gamed because of the one-per-door-per-episode guard.

---

## Try it

The environment is built on [OpenEnv](https://github.com/huggingface/openenv) and deployed as a HuggingFace Space.

*[Add HF Space link here]*

```bash
# Run locally
uv sync && uv run server

# Reset and step
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"difficulty": "medium"}'
curl -X POST http://localhost:8000/step  -H "Content-Type: application/json" -d '{"action": "move", "direction": "north"}'
```

The training notebook is at *[Add Colab link here]* — you can re-run the full GRPO training loop against a live Pyre server.

---

*Built for the OpenEnv Hackathon, India 2026.*
