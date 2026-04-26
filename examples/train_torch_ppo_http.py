"""PPO trainer that talks to the Pyre env via HTTP (localhost:8000).

Identical training logic to train_torch_ppo.py, but the environment is
accessed through the REST API instead of a direct Python import.  This
lets you run the server once and connect any number of training scripts,
remote notebooks, or evaluation tools to the same live instance.

Usage
-----
1.  Start the server (in a separate terminal):
        cd openenv-pyre
        .venv/Scripts/python.exe server/app.py

2.  Run this script:
        .venv/Scripts/python.exe examples/train_torch_ppo_http.py

Optional flags (identical to train_torch_ppo.py):
    --server          Base URL of the Pyre server  [default: http://localhost:8000]
    --episodes        Total training episodes       [default: 400]
    --difficulty-schedule  Curriculum              [default: easy,easy,easy,medium,medium]
    --output          Where to save the model .pt  [default: artifacts/pyre_ppo_http.pt]
    ...               (all other flags are the same as train_torch_ppo.py)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Resolve project root so we can import shared models regardless of CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from models import PyreAction, PyreMapState, PyreObservation
except ImportError:
    from openenv_pyre.models import PyreAction, PyreMapState, PyreObservation

# Reuse all shared utilities from the direct-import trainer
from examples.train_torch_ppo import (
    ACTION_KEYS,
    ACTION_DIM,
    ACTION_TO_INDEX,
    DIFFICULTIES,
    MAX_DOORS,
    MAX_GRID_H,
    MAX_GRID_W,
    WAIT_KEY,
    WINDS,
    ActorCritic,
    ObservationEncoder,
    RolloutBuffer,
    action_index_to_env_action,
    build_action_mask,
    compute_gae,
    ppo_update,
    save_training_graph_png,
)


# ---------------------------------------------------------------------------
# HTTP environment wrapper
# ---------------------------------------------------------------------------

class HttpPyreEnv:
    """Thin wrapper around the Pyre REST API.

    Exposes the same ``reset()`` / ``step()`` interface as ``PyreEnvironment``
    so the episode runner needs no changes.

    POST /reset  → {"difficulty": str, "seed"?: int}
    POST /step   → {"action": str, "direction"?: str,
                    "target_id"?: str, "door_state"?: str}
    Both return  → {"observation": {...}, "reward": float,
                    "done": bool, "metadata": {...}}
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    def _parse(self, data: Dict[str, Any]) -> PyreObservation:
        """Convert a raw JSON response dict into a PyreObservation."""
        obs_raw = data.get("observation", data)

        map_state: Optional[PyreMapState] = None
        ms_raw = obs_raw.get("map_state")
        if ms_raw:
            map_state = PyreMapState(**ms_raw)

        return PyreObservation(
            narrative=obs_raw.get("narrative", ""),
            agent_evacuated=obs_raw.get("agent_evacuated", False),
            location_label=obs_raw.get("location_label", ""),
            smoke_level=obs_raw.get("smoke_level", "none"),
            fire_visible=obs_raw.get("fire_visible", False),
            fire_direction=obs_raw.get("fire_direction"),
            agent_health=float(obs_raw.get("agent_health", 100.0)),
            health_status=obs_raw.get("health_status", "Good"),
            wind_dir=obs_raw.get("wind_dir", "CALM"),
            visible_objects=obs_raw.get("visible_objects", []),
            blocked_exit_ids=obs_raw.get("blocked_exit_ids", []),
            audible_signals=obs_raw.get("audible_signals", []),
            elapsed_steps=obs_raw.get("elapsed_steps", 0),
            last_action_feedback=obs_raw.get("last_action_feedback", ""),
            available_actions_hint=obs_raw.get("available_actions_hint", []),
            map_state=map_state,
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> PyreObservation:
        payload: Dict[str, Any] = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        resp = self.session.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return self._parse(resp.json())

    # ------------------------------------------------------------------
    def step(self, action: PyreAction) -> PyreObservation:
        payload: Dict[str, Any] = {"action": action.action}
        if action.direction is not None:
            payload["direction"] = action.direction
        if action.target_id is not None:
            payload["target_id"] = action.target_id
        if action.door_state is not None:
            payload["door_state"] = action.door_state
        resp = self.session.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return self._parse(resp.json())

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the server is reachable."""
        try:
            r = self.session.get(f"{self.base_url}/state", timeout=5)
            return r.status_code < 500
        except requests.exceptions.RequestException:
            return False


# ---------------------------------------------------------------------------
# Episode runner (identical reward shaping as train_torch_ppo.py)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    evacuated: bool
    final_health: float
    difficulty: str


def run_episode(
    env: HttpPyreEnv,
    network: ActorCritic,
    encoder: ObservationEncoder,
    device: torch.device,
    difficulty: str,
    history_length: int,
    buffer: RolloutBuffer,
    deterministic: bool = False,
) -> EpisodeResult:
    observation = env.reset(difficulty=difficulty)
    zero_frame = np.zeros(encoder.base_dim, dtype=np.float32)
    frames: deque = deque([zero_frame.copy() for _ in range(history_length)], maxlen=history_length)
    frames.append(encoder.encode(observation))

    total_reward = 0.0
    final_health = observation.agent_health
    evacuated = False
    steps = 0
    LOOP_WINDOW = 12
    recent_positions: deque = deque(maxlen=LOOP_WINDOW)

    network.eval()
    with torch.no_grad():
        while True:
            state_vec = np.concatenate(list(frames), dtype=np.float32)
            action_mask = build_action_mask(observation, exclude_look=True)

            obs_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

            action_t, log_prob_t, value_t = network.act(obs_t, mask_t, deterministic=deterministic)

            action_idx = int(action_t.item())
            env_action = action_index_to_env_action(action_idx)
            next_obs = env.step(env_action)

            reward = float(next_obs.reward or 0.0)
            chosen_action = env_action.action

            # Shaping 1 — idle penalty
            if chosen_action == "wait":
                reward -= 0.05

            # Shaping 2 — fire-approach penalty
            ms_next = next_obs.map_state
            if ms_next is not None and chosen_action.startswith("move"):
                ax, ay = ms_next.agent_x, ms_next.agent_y
                gw, gh = ms_next.grid_w, ms_next.grid_h
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < gw and 0 <= ny < gh:
                        if float(ms_next.fire_grid[ny * gw + nx]) > 0.15:
                            reward -= 0.15
                            break

            # Shaping 3 — anti-loop penalty
            if ms_next is not None and chosen_action.startswith("move"):
                cur_pos = (ms_next.agent_x, ms_next.agent_y)
                if cur_pos in recent_positions:
                    reward -= 0.2
                recent_positions.append(cur_pos)

            done = bool(next_obs.done)

            buffer.obs.append(state_vec)
            buffer.masks.append(action_mask)
            buffer.actions.append(action_idx)
            buffer.rewards.append(reward)
            buffer.log_probs.append(float(log_prob_t.item()))
            buffer.values.append(float(value_t.item()))
            buffer.dones.append(done)

            total_reward += reward
            steps += 1
            final_health = next_obs.agent_health
            evacuated = next_obs.agent_evacuated

            frames.append(encoder.encode(next_obs))
            observation = next_obs
            if done:
                break

    return EpisodeResult(
        total_reward=total_reward,
        steps=steps,
        evacuated=evacuated,
        final_health=final_health,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(mode=args.observation_mode)
    input_dim = encoder.base_dim * args.history_length
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    action_dim = ACTION_DIM

    # Connect to server
    env = HttpPyreEnv(base_url=args.server)
    print(f"[server] Connecting to {args.server} ...", end=" ", flush=True)
    if not env.health_check():
        print("FAILED\n[error] Server not reachable. Start it with: python server/app.py")
        sys.exit(1)
    print("OK")

    # Network
    network = ActorCritic(input_dim, action_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"\n[config] server={args.server}")
    print(f"[config] device={device}  episodes={args.episodes}  batch={args.update_every} eps")
    print(f"[config] curriculum: {args.difficulty_schedule}")
    print(f"[config] PPO clip_eps={args.clip_eps}  entropy={args.entropy_coef}  lr={args.lr}")
    print(f"\n[network] Parameters: {total_params:,}")
    print(f"[network] Input dim:  {input_dim:,}  (encoder.base_dim={encoder.base_dim} x {args.history_length} frames)")
    print(f"[network] Action dim: {action_dim}  (4 move + 4 look + 1 wait + {MAX_DOORS} open + {MAX_DOORS} close)\n", flush=True)

    schedule = args.difficulty_schedule.split(",")
    buffer = RolloutBuffer()
    metrics: list = []
    eval_metrics: list = []
    success_window: deque = deque(maxlen=30)
    reward_window: deque = deque(maxlen=30)
    t0 = time.time()
    lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.episodes
    )

    for ep in range(1, args.episodes + 1):
        stage_idx = min(int((ep - 1) / args.episodes * len(schedule)), len(schedule) - 1)
        difficulty = schedule[stage_idx]

        result = run_episode(env, network, encoder, device, difficulty, args.history_length, buffer)
        success_window.append(1 if result.evacuated else 0)
        reward_window.append(result.total_reward)
        suc30 = sum(success_window) / len(success_window)
        r30 = sum(reward_window) / len(reward_window)
        elapsed = int(time.time() - t0)

        evac_sym = "1" if result.evacuated else "0"
        print(
            f"ep={ep:04d} [{difficulty:<6}] steps={result.steps:03d}  "
            f"reward={result.total_reward:+8.3f}  evac={evac_sym}  "
            f"hp={result.final_health:5.1f}  suc30={suc30:.2f}  "
            f"r30={r30:+7.2f}  t={elapsed}s"
        )

        metrics.append({
            "episode": ep, "difficulty": difficulty, "steps": result.steps,
            "reward": round(result.total_reward, 4), "evacuated": int(result.evacuated),
            "final_health": result.final_health, "suc30": round(suc30, 3), "r30": round(r30, 3),
        })

        # PPO update
        if ep % args.update_every == 0 and len(buffer.obs) > 0:
            network.train()
            stats = ppo_update(
                network=network, optimizer=optimizer, buffer=buffer, device=device,
                clip_eps=args.clip_eps, value_clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef, value_coef=args.value_coef,
                n_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                max_grad_norm=args.max_grad_norm,
            )
            lr_scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  >> PPO update  samples=flushed  "
                f"pi_loss={stats['policy_loss']:+.4f}  v_loss={stats['value_loss']:.4f}  "
                f"entropy={stats['entropy']:.4f}  kl={stats['approx_kl']:.4f}  "
                f"clip%={stats['clip_frac']:.2f}  lr={cur_lr:.2e}"
            )
            buffer.clear()
            network.eval()

        # Evaluation
        if ep % args.eval_every == 0:
            eval_rewards, eval_success, eval_steps_list = [], [], []
            eval_buf = RolloutBuffer()
            for _ in range(args.eval_episodes):
                er = run_episode(
                    env, network, encoder, device,
                    args.eval_difficulty, args.history_length,
                    eval_buf, deterministic=True,
                )
                eval_rewards.append(er.total_reward)
                eval_success.append(1 if er.evacuated else 0)
                eval_steps_list.append(er.steps)
            avg_r = sum(eval_rewards) / len(eval_rewards)
            avg_s = sum(eval_success) / len(eval_success)
            avg_st = sum(eval_steps_list) / len(eval_steps_list)
            print(f"  ** EVAL [{args.eval_difficulty}]  reward={avg_r:+.3f}  success={avg_s:.2f}  steps={avg_st:.1f}")
            eval_metrics.append({
                "episode": ep, "eval_difficulty": args.eval_difficulty,
                "avg_reward": round(avg_r, 4), "success_rate": round(avg_s, 3),
                "avg_steps": round(avg_st, 1),
            })

        # Checkpoint
        if args.checkpoint and ep % args.checkpoint_every == 0:
            torch.save(network.state_dict(), args.checkpoint)
            print(f"  [ckpt] saved -> {args.checkpoint}")

    # --- Save artefacts ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), out)
    print(f"\n[done] Model saved -> {out}")

    if args.save_metrics and metrics:
        csv_path = out.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        print(f"[done] Metrics CSV  -> {csv_path}")

        if eval_metrics:
            eval_csv = out.with_stem(out.stem + "_eval").with_suffix(".csv")
            with open(eval_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=eval_metrics[0].keys())
                writer.writeheader()
                writer.writerows(eval_metrics)
            print(f"[done] Eval CSV     -> {eval_csv}")

    if args.save_graph:
        try:
            png_path = out.with_suffix(".png")
            save_training_graph_png(metrics, eval_metrics, str(png_path))
            print(f"[done] Graph PNG    -> {png_path}")
        except Exception as e:
            print(f"[warn] Graph skipped: {e}")

    suc_final = sum(success_window) / max(1, len(success_window))
    r_final = sum(reward_window) / max(1, len(reward_window))
    elapsed_total = time.time() - t0
    print(f"\n[summary] {args.episodes} episodes in {elapsed_total:.1f}s  ({args.episodes / elapsed_total:.1f} eps/s)")
    print(f"[summary] Final success rate (last 30): {suc_final:.2f}")
    print(f"[summary] Final reward mean  (last 30): {r_final:+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO trainer using the Pyre HTTP server (localhost:8000)"
    )

    # Server
    p.add_argument("--server", type=str, default="http://localhost:8000",
                   help="Base URL of the running Pyre env server")

    # Training
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--device", type=str, default="cpu", choices=("cuda", "cpu"))

    # Curriculum
    p.add_argument("--difficulty-schedule", type=str, default="easy,easy,easy,medium,medium")
    p.add_argument("--eval-difficulty", type=str, default="medium", choices=DIFFICULTIES)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)

    # Observation
    p.add_argument("--observation-mode", type=str, default="visible", choices=("visible", "full"))
    p.add_argument("--history-length", type=int, default=4)

    # Network
    p.add_argument("--hidden-sizes", type=str, default="256,128,64")

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--update-every", type=int, default=5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Output
    p.add_argument("--output", type=str, default="artifacts/pyre_ppo_http.pt")
    p.add_argument("--checkpoint", type=str, default="artifacts/pyre_ppo_http_ckpt.pt")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--save-metrics", action="store_true", default=True)
    p.add_argument("--save-graph", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
