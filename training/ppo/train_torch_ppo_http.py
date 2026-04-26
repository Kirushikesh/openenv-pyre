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

Optional flags (all match train_torch_ppo.py):
    --server               Base URL of the Pyre server   [default: http://localhost:8000]
    --episodes             Total training episodes        [default: 400]
    --difficulty-schedule  Curriculum stages              [default: easy,medium,hard]
    --patience-threshold   Success-rate gate (0=static)  [default: 0.65]
    --learning-rate        Adam learning rate             [default: 3e-4]
    --resume               Path to checkpoint to resume  [default: None]
    --output               Where to save the model .pt   [default: artifacts/pyre_ppo_http.pt]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# TeeLogger — mirrors all stdout output to a log file simultaneously
# ---------------------------------------------------------------------------

class TeeLogger:
    """Writes every print() to both the original stdout and a log file."""

    def __init__(self, stream, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = stream
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)

# ---------------------------------------------------------------------------
# Resolve project root so we can import shared models regardless of CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from models import PyreAction, PyreMapState, PyreObservation
except ImportError:
    from openenv_pyre.models import PyreAction, PyreMapState, PyreObservation

# Reuse all shared utilities from the direct-import trainer
from training.ppo.train_torch_ppo import (
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
    PatienceCurriculum,
    RolloutBuffer,
    action_index_to_env_action,
    build_action_mask,
    build_curriculum,
    compute_gae,
    parse_mix_dist,
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
    step_delay: float = 0.0,
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
            if step_delay > 0.0:
                time.sleep(step_delay)
            print(f"    step={steps+1:03d}  action={ACTION_KEYS[action_idx]:<40}  hp={observation.agent_health:5.1f}", flush=True)
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

            # Shaping 4 — exit proximity pull
            # Absolute distance-based bonus (not just delta) so the network
            # has a continuous gradient toward exits from anywhere on the map.
            # Max +0.25 when adjacent, tapers to 0 beyond 6 cells (Manhattan).
            if ms_next is not None and chosen_action.startswith("move") and not next_obs.agent_evacuated:
                ax, ay = ms_next.agent_x, ms_next.agent_y
                exits = ms_next.exit_positions
                if exits:
                    min_manhattan = min(abs(ax - ex[0]) + abs(ay - ex[1]) for ex in exits)
                    reward += max(0.0, 0.25 - 0.04 * min_manhattan)

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
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available — falling back to CPU.")

    encoder = ObservationEncoder(mode=args.observation_mode)
    input_dim = encoder.base_dim * args.history_length
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))
    action_dim = ACTION_DIM

    # Connect to server
    env = HttpPyreEnv(base_url=args.server)
    print(f"[server] Connecting to {args.server} ...", end=" ", flush=True)
    if not env.health_check():
        print("FAILED\n[error] Server not reachable. Start it with: python server/app.py")
        sys.exit(1)
    print("OK")

    # Network + optimizer
    network = ActorCritic(input_dim, action_dim, hidden_sizes).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate, eps=1e-5)

    # LinearLR scheduler: step once per PPO update, not per episode
    total_updates = args.episodes // args.update_every
    lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args.lr_end_factor,
        total_iters=max(1, total_updates),
    ) if args.lr_decay else None

    total_params = sum(p.numel() for p in network.parameters())
    print(f"\n[config] server={args.server}")
    print(f"[config] device={device}  episodes={args.episodes}  batch={args.update_every} eps")
    print(f"[config] curriculum: {args.difficulty_schedule}")
    print(f"[config] PPO clip_eps={args.clip_eps}  entropy={args.entropy_coef}  lr={args.learning_rate}")
    print(f"\n[network] Parameters: {total_params:,}")
    print(f"[network] Input dim:  {input_dim:,}  (encoder.base_dim={encoder.base_dim} x {args.history_length} frames)")
    print(f"[network] Action dim: {action_dim}  (4 move + 4 look + 1 wait + {MAX_DOORS} open + {MAX_DOORS} close)\n", flush=True)

    # Build curriculum — patience-gated (dynamic) or static
    stages = [s.strip().lower() for s in args.difficulty_schedule.split(",") if s.strip()]
    if args.patience_threshold > 0:
        mix_dist = parse_mix_dist(getattr(args, "hard_mix_dist", None))
        patience_curriculum = PatienceCurriculum(
            stages=stages,
            threshold=args.patience_threshold,
            patience_window=args.patience_window,
            mix_ratio=args.hard_mix_ratio,
            mix_dist=mix_dist,
        )
        static_schedule: Optional[List[str]] = None
        print(f"[curriculum] patience-gated: threshold={args.patience_threshold}  "
              f"window={args.patience_window}  mix={args.hard_mix_ratio}", flush=True)
        if mix_dist is not None:
            print(f"[curriculum] hard-phase mix distribution: {mix_dist}", flush=True)
    else:
        patience_curriculum = None
        static_schedule = build_curriculum(args.difficulty_schedule, args.episodes)
        print(f"[curriculum] static: {args.difficulty_schedule}", flush=True)

    # Resume
    start_ep = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        network.load_state_dict(ckpt.get("network_state", ckpt))
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if lr_scheduler and ckpt.get("scheduler_state"):
            lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        start_ep = int(ckpt.get("episode", 0))
        print(f"[resume] Loaded checkpoint from episode {start_ep}: {args.resume}")

    buffer = RolloutBuffer()
    episode_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    success_window: deque = deque(maxlen=30)
    reward_window: deque = deque(maxlen=30)
    t0 = time.time()

    for ep_idx in range(start_ep, args.episodes):
        ep = ep_idx + 1

        # Determine difficulty for this episode
        if patience_curriculum is not None:
            difficulty = patience_curriculum.current
        else:
            difficulty = static_schedule[ep_idx]  # type: ignore[index]

        # Use step delay only after --viz-after-ep episodes have been trained
        ep_step_delay = args.step_delay if ep > args.viz_after_ep else 0.0
        result = run_episode(env, network, encoder, device, difficulty, args.history_length, buffer,
                             step_delay=ep_step_delay)

        success_window.append(1 if result.evacuated else 0)
        reward_window.append(result.total_reward)
        suc30 = sum(success_window) / len(success_window)
        r30 = sum(reward_window) / len(reward_window)
        elapsed = int(time.time() - t0)

        # Advance patience curriculum after updating success_window
        if patience_curriculum is not None:
            difficulty = patience_curriculum.step(suc30)

        print(
            f"ep={ep:04d} [{difficulty:<6}] steps={result.steps:03d}  "
            f"reward={result.total_reward:+8.3f}  evac={int(result.evacuated)}  "
            f"hp={result.final_health:5.1f}  suc30={suc30:.2f}  "
            f"r30={r30:+7.2f}  t={elapsed}s",
            flush=True,
        )

        episode_rows.append({
            "episode": ep,
            "difficulty": difficulty,
            "steps": result.steps,
            "reward": round(result.total_reward, 4),
            "evacuated": int(result.evacuated),
            "final_health": round(result.final_health, 2),
            "reward_mean_30": round(r30, 4),
            "success_rate_30": round(suc30, 4),
        })

        # PPO update every N episodes (or at the very last episode)
        should_update = (ep % args.update_every == 0) or (ep == args.episodes)
        if should_update and len(buffer) > 0:
            network.train()
            stats = ppo_update(
                network=network, optimizer=optimizer, buffer=buffer, device=device,
                clip_eps=args.clip_eps, value_clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef, value_coef=args.value_coef,
                n_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                max_grad_norm=args.max_grad_norm,
            )
            if lr_scheduler:
                lr_scheduler.step()
            buffer.clear()
            network.eval()

            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  >> PPO update  samples=flushed  "
                f"pi_loss={stats['policy_loss']:+.4f}  v_loss={stats['value_loss']:.4f}  "
                f"entropy={stats['entropy']:.4f}  kl={stats['approx_kl']:.4f}  "
                f"clip%={stats['clip_frac']:.2f}  lr={cur_lr:.2e}",
                flush=True,
            )

        # Evaluation
        if args.eval_every > 0 and (ep % args.eval_every == 0 or ep == args.episodes):
            eval_rewards, eval_success, eval_steps_list = [], [], []
            eval_buf = RolloutBuffer()
            for _ in range(args.eval_episodes):
                er = run_episode(
                    env, network, encoder, device,
                    args.eval_difficulty, args.history_length,
                    eval_buf, deterministic=True, step_delay=0.0,
                )
                eval_buf.clear()  # clear after each eval episode — don't accumulate
                eval_rewards.append(er.total_reward)
                eval_success.append(1 if er.evacuated else 0)
                eval_steps_list.append(er.steps)
            avg_r = sum(eval_rewards) / len(eval_rewards)
            avg_s = sum(eval_success) / len(eval_success)
            avg_st = sum(eval_steps_list) / len(eval_steps_list)
            print(f"  ** EVAL [{args.eval_difficulty}]  reward={avg_r:+.3f}  success={avg_s:.2f}  steps={avg_st:.1f}", flush=True)
            eval_rows.append({
                "episode": ep,
                "difficulty": args.eval_difficulty,
                "reward_mean": round(avg_r, 4),
                "success_rate": round(avg_s, 3),
                "steps_mean": round(avg_st, 1),
            })

        # Periodic checkpoint (full state, same as train_torch_ppo.py)
        if args.checkpoint and args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            ckpt_path = Path(args.checkpoint)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "episode": ep,
                "network_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
                "args": vars(args),
            }, ckpt_path)
            print(f"  [ckpt] saved -> {args.checkpoint}", flush=True)

    # --- Save artefacts ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episode": args.episodes,
        "network_state": network.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
        "args": vars(args),
    }, out)
    print(f"\n[done] Model saved -> {out}")

    if args.save_metrics and episode_rows:
        csv_path = out.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"[done] Metrics CSV  -> {csv_path}")

        if eval_rows:
            eval_csv = out.parent / (out.stem + "_eval.csv")
            with open(eval_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
                writer.writeheader()
                writer.writerows(eval_rows)
            print(f"[done] Eval CSV     -> {eval_csv}")

    if args.save_graph:
        png_path = out.with_suffix(".png")
        # Correct arg order: save_training_graph_png(path, episode_rows, eval_rows)
        save_training_graph_png(png_path, episode_rows, eval_rows)
        print(f"[done] Graph PNG    -> {png_path}")

    suc_final = sum(success_window) / max(1, len(success_window))
    r_final = sum(reward_window) / max(1, len(reward_window))
    elapsed_total = time.time() - t0
    n_trained = args.episodes - start_ep
    print(f"\n[summary] {n_trained} episodes in {elapsed_total:.1f}s  ({n_trained / max(1, elapsed_total):.1f} eps/s)")
    print(f"[summary] Final success rate (last 30): {suc_final:.2f}")
    print(f"[summary] Final reward mean  (last 30): {r_final:+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO trainer using the Pyre HTTP server (localhost:8000)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    p.add_argument("--server", type=str, default="http://localhost:8000",
                   help="Base URL of the running Pyre env server")

    # Visualization / pacing
    p.add_argument("--step-delay", type=float, default=0.0,
                   help="Seconds to sleep between steps (0=full speed, 0.5=smooth viz)")
    p.add_argument("--viz-after-ep", type=int, default=0,
                   help="Episode after which --step-delay activates. "
                        "0=always delay, 100=fast first 100 eps then slow.")

    # Training scale
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=150, help="Max steps per episode (informational; enforced server-side)")
    p.add_argument("--device", type=str, default="cpu", choices=("cuda", "cpu"))

    # Curriculum
    p.add_argument("--difficulty-schedule", type=str, default="easy,medium,hard",
                   help="Comma-separated curriculum stages")
    p.add_argument("--patience-threshold", type=float, default=0.65,
                   help="Success-rate (30-ep window) required before advancing difficulty. Set 0 for static split.")
    p.add_argument("--patience-window", type=int, default=15,
                   help="Consecutive episodes that must meet --patience-threshold before advancing.")
    p.add_argument("--hard-mix-dist", type=str, default=None,
                   help="Cumulative replay distribution for the final stage, e.g. "
                        "'hard:0.6,medium:0.3,easy:0.1'. Overrides --hard-mix-ratio.")
    p.add_argument("--hard-mix-ratio", type=float, default=0.25,
                   help="Fraction of hard-phase episodes replayed on medium (prevents forgetting).")
    p.add_argument("--eval-difficulty", type=str, default="medium", choices=DIFFICULTIES)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)

    # Observation
    p.add_argument("--observation-mode", type=str, default="visible", choices=("visible", "full"))
    p.add_argument("--history-length", type=int, default=4)

    # Network
    p.add_argument("--hidden-sizes", type=str, default="512,256,128",
                   help="Comma-separated MLP hidden layer sizes (match train_torch_ppo.py defaults)")

    # PPO hyperparameters
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-decay", action="store_true", default=True,
                   help="Linear LR decay to lr-end-factor × initial LR over training updates")
    p.add_argument("--lr-end-factor", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--update-every", type=int, default=5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Persistence
    p.add_argument("--output", type=str, default="artifacts/pyre_ppo_http.pt")
    p.add_argument("--checkpoint", type=str, default="artifacts/pyre_ppo_http_ckpt.pt")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint (.pt) to resume training from")
    p.add_argument("--save-metrics", action="store_true", default=True)
    p.add_argument("--save-graph", action="store_true", default=True)
    p.add_argument("--log-file", type=str, default=None,
                   help="Path to write a copy of all console output (e.g. artifacts/train_http.log). "
                        "Output is written to both the terminal and the file simultaneously.")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_tee = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_tee = TeeLogger(sys.stdout, log_path)
        sys.stdout = log_tee  # type: ignore[assignment]
        print(f"[log] Writing console output to {log_path}", flush=True)

    try:
        train(args)
    finally:
        if log_tee:
            sys.stdout = log_tee._stream
            log_tee.close()
            print(f"[log] Training log saved -> {args.log_file}")


if __name__ == "__main__":
    main()
