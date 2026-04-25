"""Random-action baseline agent for Pyre (single-agent).

Runs N episodes using the PyreEnv client and prints per-episode stats.
Use this to smoke-test the server and verify the reward distribution
spans a meaningful range.

Usage:
    # Server must be running first:
    #   cd pyre_env && uv run server
    #
    python examples/random_agent.py --episodes 5
    python examples/random_agent.py --episodes 5 --verbose
    python examples/random_agent.py --url http://localhost:8000 --episodes 10
"""

import argparse
import random
import sys
from typing import List

import requests

from pyre_env import PyreEnv, PyreAction


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def _parse_hint(hint: str) -> PyreAction:
    """Parse a hint string from available_actions_hint into a PyreAction."""
    try:
        h = hint.strip()
        if h.startswith("move("):
            return PyreAction(action="move", direction=h.split("'")[1])
        elif h.startswith("door("):
            parts = h.split("'")
            # parts: ["door(target_id=", did, ", door_state=", state, ")"]
            target_id = parts[1]
            door_state = parts[3]
            return PyreAction(action="door", target_id=target_id, door_state=door_state)
        elif h == "wait()":
            return PyreAction(action="wait")
    except (IndexError, ValueError):
        pass
    return PyreAction(action="wait")


def random_action(hints: List[str], rng: random.Random) -> PyreAction:
    """Pick a random action, biasing toward available hints 70% of the time."""
    if hints and rng.random() < 0.7:
        return _parse_hint(rng.choice(hints))
    # Fallback: random move
    return PyreAction(action="move", direction=rng.choice(["north", "south", "east", "west"]))


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, max_steps: int, rng: random.Random, verbose: bool) -> dict:
    result = env.reset()
    obs = result.observation

    episode_reward = 0.0
    steps = 0
    done = result.done

    while not done and steps < max_steps:
        action = random_action(obs.available_actions_hint, rng)
        result = env.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done
        episode_reward += reward
        steps += 1

        if verbose:
            first_line = obs.narrative.split("\n")[0] if obs.narrative else ""
            print(
                f"  step {steps:3d} | hp={obs.agent_health:5.1f}"
                f" | r={reward:+.3f} | done={done} | {first_line[:70]}"
            )

    meta = obs.metadata or {}
    return {
        "steps": steps,
        "total_reward": round(episode_reward, 3),
        "done": done,
        "evacuated": obs.agent_evacuated,
        "final_health": obs.agent_health,
        "wind_dir": obs.wind_dir,
        "fire_sources": meta.get("fire_sources", "?"),
        "fire_spread": meta.get("fire_spread_rate", "?"),
        "last_narrative": obs.narrative[:120] if obs.narrative else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pyre random-agent baseline")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        r.raise_for_status()
        print(f"Server healthy: {args.url}")
    except Exception as e:
        print(f"Server not reachable at {args.url}: {e}")
        sys.exit(1)

    rng = random.Random(args.seed)
    results: List[dict] = []

    with PyreEnv(base_url=args.url).sync() as env:
        for ep in range(args.episodes):
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            stats = run_episode(env, args.max_steps, rng, args.verbose)
            results.append(stats)
            print(
                f"  DONE  steps={stats['steps']}  reward={stats['total_reward']:+.3f}"
                f"  health={stats['final_health']:.1f}"
                f"  wind={stats['wind_dir']}  sources={stats['fire_sources']}"
                f"  spread={stats['fire_spread']}"
            )

    print("\n=== Summary ===")
    rewards = [r["total_reward"] for r in results]
    print(f"Episodes:       {len(results)}")
    print(f"Reward min/max: {min(rewards):.3f} / {max(rewards):.3f}")
    print(f"Reward mean:    {sum(rewards)/len(rewards):.3f}")
    print(f"Avg steps:      {sum(r['steps'] for r in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
