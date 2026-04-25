"""Random-action baseline agent for Pyre.

Runs N episodes with random actions and prints per-episode stats.
Use this to smoke-test the server and verify the reward distribution
spans a meaningful range.

Usage:
    # Server must be running first:
    #   uv run server   (from pyre_env/ directory)
    #
    python examples/random_agent.py --episodes 5 --max-steps 50
    python examples/random_agent.py --url http://localhost:8000
"""

import argparse
import random
import sys
import time
from typing import List

import requests


# ---------------------------------------------------------------------------
# Action space (must match server-side PyreAction)
# ---------------------------------------------------------------------------

MOVE_DIRS = ["north", "south", "east", "west"]
BROADCAST_CATEGORIES = [
    "evacuate_north", "evacuate_south", "evacuate_east", "evacuate_west",
    "stay_calm", "use_exit_1", "use_exit_2",
]


def random_action(obs: dict, rng: random.Random) -> dict:
    """Build a random but structurally valid action from the current observation."""
    action_hints = obs.get("available_actions_hint", [])

    # 60% of the time: follow an available action hint
    if action_hints and rng.random() < 0.6:
        hint = rng.choice(action_hints)
        parsed = _parse_hint(hint)
        if parsed:
            return parsed

    # Fallback: random move
    return {"action": "move", "direction": rng.choice(MOVE_DIRS)}


def _parse_hint(hint: str) -> dict:
    """Parse a hint string like move(direction='north') into an action dict."""
    try:
        hint = hint.strip()
        if hint.startswith("move("):
            # move(direction='north')
            direction = hint.split("'")[1]
            return {"action": "move", "direction": direction}
        elif hint.startswith("close_door("):
            door_id = hint.split("'")[1]
            return {"action": "close_door", "target_id": door_id}
        elif hint.startswith("open_door("):
            door_id = hint.split("'")[1]
            return {"action": "open_door", "target_id": door_id}
        elif hint.startswith("instruct("):
            # instruct(target_id='p_1', direction='south')
            parts = hint.split("'")
            target_id = parts[1]
            direction = parts[3]
            return {"action": "instruct", "target_id": target_id, "direction": direction}
        elif hint.startswith("broadcast("):
            parts = hint.split("'")
            zone = parts[1]
            category = parts[3]
            return {"action": "broadcast", "zone": zone, "category": category}
        elif hint == "wait()":
            return {"action": "wait"}
    except (IndexError, ValueError):
        pass
    return {}


# ---------------------------------------------------------------------------
# HTTP helpers (no client library needed)
# ---------------------------------------------------------------------------

def api_reset(base_url: str) -> dict:
    r = requests.post(f"{base_url}/reset", timeout=10)
    r.raise_for_status()
    return r.json()


def api_step(base_url: str, action: dict) -> dict:
    r = requests.post(f"{base_url}/step", json=action, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_episode(base_url: str, max_steps: int, rng: random.Random, verbose: bool) -> dict:
    result = api_reset(base_url)
    obs = result.get("observation", result)

    episode_reward = 0.0
    steps = 0
    done = result.get("done", False)

    while not done and steps < max_steps:
        action = random_action(obs, rng)
        result = api_step(base_url, action)

        obs = result.get("observation", result)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        episode_reward += reward
        steps += 1

        if verbose:
            narrative = obs.get("narrative", "")
            first_line = narrative.split("\n")[0] if narrative else ""
            print(f"  step {steps:3d} | r={reward:+.3f} | done={done} | {first_line[:80]}")

    meta = result.get("metadata", {})
    return {
        "steps": steps,
        "total_reward": round(episode_reward, 3),
        "done": done,
        "npcs_evacuated": meta.get("npcs_evacuated", "?"),
        "npcs_casualties": meta.get("npcs_casualties", "?"),
        "stampede_events": meta.get("stampede_events", "?"),
        "last_narrative": obs.get("narrative", "")[:120],
    }


def main():
    parser = argparse.ArgumentParser(description="Pyre random-agent baseline")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=80, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--verbose", action="store_true", help="Print each step")
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

    for ep in range(args.episodes):
        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
        stats = run_episode(args.url, args.max_steps, rng, args.verbose)
        results.append(stats)
        print(
            f"  DONE  steps={stats['steps']}  reward={stats['total_reward']:+.3f}"
            f"  evac={stats['npcs_evacuated']}  casualties={stats['npcs_casualties']}"
            f"  stampedes={stats['stampede_events']}"
        )

    print("\n=== Summary ===")
    rewards = [r["total_reward"] for r in results]
    print(f"Episodes:       {len(results)}")
    print(f"Reward min/max: {min(rewards):.3f} / {max(rewards):.3f}")
    print(f"Reward mean:    {sum(rewards)/len(rewards):.3f}")
    print(f"Avg steps:      {sum(r['steps'] for r in results) / len(results):.1f}")


if __name__ == "__main__":
    main()
