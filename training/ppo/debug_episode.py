"""
Debug a single Pyre episode using a trained PPO checkpoint.

Runs one episode on the specified difficulty, prints a rich step-by-step
trace, and produces a failure analysis at the end.

Usage
-----
  # Debug a hard episode with the latest checkpoint
  uv run python training/ppo/debug_episode.py \
      --checkpoint artifacts/pyre_ppo_hard_ckpt.pt \
      --difficulty hard

  # Override difficulty or seed for reproducibility
  uv run python training/ppo/debug_episode.py \
      --checkpoint artifacts/pyre_ppo_hard_ckpt.pt \
      --difficulty hard \
      --seed 7

  # Stochastic policy (not deterministic) to see exploration behaviour
  uv run python training/ppo/debug_episode.py \
      --checkpoint artifacts/pyre_ppo_hard_ckpt.pt \
      --difficulty hard \
      --stochastic
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.ppo.train_torch_ppo import (
    ACTION_KEYS,
    ACTION_DIM,
    ActorCritic,
    ObservationEncoder,
    RolloutBuffer,
    action_index_to_env_action,
    build_action_mask,
)
from training.ppo.train_torch_ppo_http import HttpPyreEnv

# ── ANSI colours (degrade gracefully on Windows) ──────────────────────────────
try:
    import os
    _COLOUR = os.get_terminal_size().columns > 0
except Exception:
    _COLOUR = False

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

RED    = lambda t: _c(t, "31")
YELLOW = lambda t: _c(t, "33")
GREEN  = lambda t: _c(t, "32")
CYAN   = lambda t: _c(t, "36")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ── Mini ASCII map renderer ───────────────────────────────────────────────────

_CELL_CHAR = {0: ".", 1: "#", 2: "/", 3: "+", 4: "E", 5: "X"}
_CELL_NAME = {0: "floor", 1: "wall", 2: "door(open)", 3: "door(closed)", 4: "EXIT", 5: "obstacle"}

def render_ascii_map(ms, visible_set: set, step: int) -> str:
    """Render a small ASCII map centered on the agent (15×15 window)."""
    ax, ay = ms.agent_x, ms.agent_y
    gw, gh = ms.grid_w, ms.grid_h
    W = 15
    x0 = max(0, ax - W // 2)
    y0 = max(0, ay - W // 2)
    x1 = min(gw, x0 + W)
    y1 = min(gh, y0 + W)

    lines = [f"  Map window [{x0}:{x1}, {y0}:{y1}]  agent=({ax},{ay})"]
    for y in range(y0, y1):
        row = []
        for x in range(x0, x1):
            idx = y * gw + x
            if x == ax and y == ay:
                row.append(BOLD(CYAN("@")))
                continue
            ct = int(ms.cell_grid[idx])
            fire = float(ms.fire_grid[idx])
            smoke = float(ms.smoke_grid[idx])
            ch = _CELL_CHAR.get(ct, "?")
            if (x, y) not in visible_set:
                row.append(DIM("·"))
            elif ct == 4:
                row.append(GREEN(ch))
            elif fire >= 0.5:
                row.append(RED("F"))
            elif fire > 0.1:
                row.append(YELLOW("f"))
            elif smoke >= 0.5:
                row.append(YELLOW("s"))
            else:
                row.append(ch)
        lines.append("  " + " ".join(row))
    return "\n".join(lines)


# ── Per-step diagnostics ──────────────────────────────────────────────────────

def _nearest_exit(ms) -> Tuple[int, Optional[List]]:
    ax, ay = ms.agent_x, ms.agent_y
    gw = ms.grid_w
    exits = ms.exit_positions
    if not exits:
        return 9999, None
    dists = [(abs(ax - e[0]) + abs(ay - e[1]), e) for e in exits]
    dists.sort()
    return dists[0]


def _fire_summary(ms) -> str:
    """Fraction of grid cells that are actively burning."""
    total = ms.grid_w * ms.grid_h
    burning = sum(1 for v in ms.fire_grid if float(v) >= 0.3)
    blocked = [e for e in ms.exit_positions if float(ms.fire_grid[e[1] * ms.grid_w + e[0]]) >= 0.5]
    return (f"burning={burning}/{total} ({100*burning/total:.1f}%)  "
            f"exits_blocked={len(blocked)}/{len(ms.exit_positions)}")


def _health_bar(hp: float) -> str:
    filled = int(hp / 5)
    bar = "█" * filled + "░" * (20 - filled)
    colour = GREEN if hp > 60 else (YELLOW if hp > 30 else RED)
    return colour(f"[{bar}] {hp:.1f} HP")


# ── Main debug runner ─────────────────────────────────────────────────────────

def debug_episode(
    checkpoint_path: str,
    difficulty: str,
    server: str,
    deterministic: bool,
    history_length: int,
    hidden_sizes: Tuple[int, ...],
    seed: Optional[int],
    show_map: bool,
) -> None:
    device = torch.device("cpu")
    encoder = ObservationEncoder(mode="visible")
    input_dim = encoder.base_dim * history_length

    network = ActorCritic(input_dim, ACTION_DIM, hidden_sizes).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(ckpt.get("network_state", ckpt))
    network.eval()
    trained_eps = ckpt.get("episode", "?")
    print(BOLD(f"\n{'='*70}"))
    print(BOLD(f"  Pyre Episode Debugger"))
    print(f"  checkpoint : {checkpoint_path}  (trained {trained_eps} eps)")
    print(f"  difficulty : {difficulty}  |  deterministic={deterministic}")
    print(f"  seed       : {seed if seed is not None else 'random'}")
    print(BOLD(f"{'='*70}\n"))

    env = HttpPyreEnv(base_url=server)
    if not env.health_check():
        print(RED("[error] Server not reachable. Start it with: uv run server"))
        sys.exit(1)

    observation = env.reset(difficulty=difficulty, seed=seed)
    ms0 = observation.map_state
    print(BOLD(f"── Episode Start ──────────────────────────────────────────────────────"))
    print(f"  Map size   : {ms0.grid_w}×{ms0.grid_h}  template={ms0.template_name}")
    print(f"  Max steps  : {ms0.max_steps}")
    print(f"  Fire spread: {ms0.fire_spread_rate:.3f}  humidity={ms0.humidity:.3f}  wind={ms0.wind_dir}")
    d, nearest = _nearest_exit(ms0)
    print(f"  Agent spawn: ({ms0.agent_x}, {ms0.agent_y})  nearest_exit={nearest}  dist={d}")
    print(f"  Exits      : {ms0.exit_positions}")
    print()

    zero_frame = np.zeros(encoder.base_dim, dtype=np.float32)
    frames: deque = deque([zero_frame.copy() for _ in range(history_length)], maxlen=history_length)
    frames.append(encoder.encode(observation))

    # ── Tracking ──
    step_log = []          # list of dicts, one per step
    loop_positions: deque = deque(maxlen=12)
    visited: set = set()
    total_reward = 0.0
    prev_exit_dist = d

    # ── Episode loop ──
    with torch.no_grad():
        while True:
            state_vec = np.concatenate(list(frames), dtype=np.float32)
            action_mask = build_action_mask(observation, exclude_look=True)

            obs_t  = torch.tensor(state_vec,  dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

            action_t, log_prob_t, value_t = network.act(obs_t, mask_t, deterministic=deterministic)
            action_idx  = int(action_t.item())
            log_prob    = float(log_prob_t.item())
            value_est   = float(value_t.item())
            env_action  = action_index_to_env_action(action_idx)

            next_obs = env.step(env_action)
            ms = next_obs.map_state
            reward = float(next_obs.reward or 0.0)
            done   = bool(next_obs.done)

            # Compute diagnostics
            exit_dist, nearest_exit = _nearest_exit(ms) if ms else (9999, None)
            exit_delta = prev_exit_dist - exit_dist   # positive = moved closer
            prev_exit_dist = exit_dist

            cur_pos = (ms.agent_x, ms.agent_y) if ms else None
            is_loop = cur_pos in loop_positions if cur_pos else False
            is_new  = cur_pos not in visited if cur_pos else False
            if cur_pos:
                loop_positions.append(cur_pos)
                visited.add(cur_pos)

            visible_set = {tuple(v) for v in (ms.visible_cells if ms else [])}
            fire_nearby = False
            if ms and cur_pos:
                gw = ms.grid_w
                for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                    nx, ny = cur_pos[0]+dx, cur_pos[1]+dy
                    if 0 <= nx < ms.grid_w and 0 <= ny < ms.grid_h:
                        if float(ms.fire_grid[ny*gw+nx]) > 0.15:
                            fire_nearby = True
                            break

            total_reward += reward
            step = len(step_log) + 1

            # Classify this step
            if next_obs.agent_evacuated:
                flag = GREEN("✓ EVACUATED")
            elif not next_obs.agent_health > 0 or (done and not next_obs.agent_evacuated):
                death_cause = "DIED" if not (ms and ms.agent_alive) else "TIMEOUT"
                flag = RED(f"✗ {death_cause}")
            elif is_loop:
                flag = YELLOW("⚠ LOOP")
            elif fire_nearby:
                flag = YELLOW("⚠ fire-adj")
            else:
                flag = ""

            step_log.append({
                "step": step,
                "action": ACTION_KEYS[action_idx],
                "hp": next_obs.agent_health,
                "reward": reward,
                "exit_dist": exit_dist,
                "exit_delta": exit_delta,
                "value_est": value_est,
                "log_prob": log_prob,
                "is_loop": is_loop,
                "is_new": is_new,
                "fire_nearby": fire_nearby,
                "smoke": next_obs.smoke_level,
                "flag": flag,
                "pos": cur_pos,
                "fire_summary": _fire_summary(ms) if ms else "",
            })

            # Print step line
            delta_str = f"{exit_delta:+.0f}" if exit_delta != 0 else " 0"
            loop_warn = YELLOW(" [LOOP]") if is_loop else ""
            fire_warn = YELLOW(" [fire-adj]") if fire_nearby else ""
            print(
                f"  step {step:03d} | {ACTION_KEYS[action_idx]:<38} | "
                f"hp={next_obs.agent_health:5.1f} | "
                f"reward={reward:+6.3f} | "
                f"dist={exit_dist:3d}({delta_str}) | "
                f"val={value_est:+5.2f} | "
                f"smoke={next_obs.smoke_level:<8} "
                f"{flag}{loop_warn}{fire_warn}",
                flush=True,
            )

            # Every 10 steps print fire state + optional map
            if step % 10 == 0:
                print(DIM(f"         fire: {step_log[-1]['fire_summary']}"))
                if show_map and ms:
                    print(render_ascii_map(ms, visible_set, step))
                print()

            frames.append(encoder.encode(next_obs))
            observation = next_obs
            if done:
                break

    # ── Post-episode analysis ──────────────────────────────────────────────────
    print(BOLD(f"\n{'='*70}"))
    print(BOLD("  Post-Episode Analysis"))
    print(BOLD(f"{'='*70}"))

    outcome = GREEN("EVACUATED ✓") if step_log[-1]["flag"].strip().endswith("EVACUATED ✓") or observation.agent_evacuated \
              else RED("FAILED ✗")
    print(f"\n  Outcome       : {outcome}")
    print(f"  Total steps   : {len(step_log)}")
    print(f"  Total reward  : {total_reward:+.3f}")
    print(f"  Final HP      : {_health_bar(observation.agent_health)}")

    # Cause of failure
    if not observation.agent_evacuated:
        ms_final = observation.map_state
        if ms_final and not ms_final.agent_alive:
            print(f"\n  {RED('Cause of failure:')} Agent DIED (health depleted by smoke/fire)")
        elif len(step_log) >= (ms_final.max_steps if ms_final else 100):
            print(f"\n  {YELLOW('Cause of failure:')} TIMEOUT — ran out of steps")
        else:
            print(f"\n  {YELLOW('Cause of failure:')} Unknown / server reported done")

    # Loop analysis
    loop_steps = [s for s in step_log if s["is_loop"]]
    print(f"\n  Loop steps    : {len(loop_steps)} / {len(step_log)}  "
          f"({100*len(loop_steps)/max(1,len(step_log)):.1f}%)")
    if loop_steps:
        positions_looped = [s["pos"] for s in loop_steps]
        from collections import Counter
        top = Counter(positions_looped).most_common(3)
        print(f"  Most-looped positions: {top}")

    # Health drain
    hp_start = 100.0
    hp_end   = observation.agent_health
    drain_per_step = (hp_start - hp_end) / max(1, len(step_log))
    print(f"\n  HP drain      : {hp_start:.0f} → {hp_end:.1f}  ({drain_per_step:.2f} HP/step avg)")

    # Reward breakdown
    positive_steps = [s for s in step_log if s["reward"] > 0]
    negative_steps = [s for s in step_log if s["reward"] < 0]
    worst_5 = sorted(step_log, key=lambda s: s["reward"])[:5]
    best_5  = sorted(step_log, key=lambda s: s["reward"], reverse=True)[:5]
    print(f"\n  Positive reward steps : {len(positive_steps)}")
    print(f"  Negative reward steps : {len(negative_steps)}")
    print(f"\n  {RED('5 worst steps')}:")
    for s in worst_5:
        print(f"    step {s['step']:03d}  reward={s['reward']:+.3f}  "
              f"action={s['action']:<35}  dist={s['exit_dist']}  hp={s['hp']:.1f}")
    print(f"\n  {GREEN('5 best steps')}:")
    for s in best_5:
        print(f"    step {s['step']:03d}  reward={s['reward']:+.3f}  "
              f"action={s['action']:<35}  dist={s['exit_dist']}  hp={s['hp']:.1f}")

    # Exit distance trend
    dists = [s["exit_dist"] for s in step_log]
    print(f"\n  Exit distance: start={dists[0]}  min={min(dists)}  "
          f"final={dists[-1]}  never_below_5={'YES' if min(dists) < 5 else 'NO'}")

    # Fire at end
    ms_final = observation.map_state
    if ms_final:
        print(f"\n  Final fire state: {_fire_summary(ms_final)}")
        blocked_at_end = [e for e in ms_final.exit_positions
                          if float(ms_final.fire_grid[e[1]*ms_final.grid_w+e[0]]) >= 0.5]
        if blocked_at_end:
            print(f"  {RED('All exits were fire-blocked at episode end!')} {blocked_at_end}")
        else:
            open_exits = [e for e in ms_final.exit_positions
                          if float(ms_final.fire_grid[e[1]*ms_final.grid_w+e[0]]) < 0.5]
            print(f"  Open exits at end: {open_exits}")

    # Diagnosis
    print(BOLD(f"\n{'─'*70}"))
    print(BOLD("  Diagnosis"))
    print(BOLD(f"{'─'*70}"))
    diags = []

    if len(loop_steps) / max(1, len(step_log)) > 0.15:
        diags.append(RED("⚠ Heavy looping") +
                     f" — agent revisited positions on {len(loop_steps)} steps. "
                     "Policy is stuck in a local movement pattern.")

    if drain_per_step > 2.5:
        diags.append(RED("⚠ Rapid health drain") +
                     f" — {drain_per_step:.1f} HP/step average. Agent is spending "
                     "too many steps inside smoke/fire zones.")

    if min(dists) >= 10:
        diags.append(RED("⚠ Never got close to exit") +
                     f" — minimum exit distance was {min(dists)}. "
                     "Agent may be exploring in the wrong direction.")

    if ms_final:
        blocked_exits = [e for e in ms_final.exit_positions
                         if float(ms_final.fire_grid[e[1]*ms_final.grid_w+e[0]]) >= 0.5]
        if len(blocked_exits) == len(ms_final.exit_positions):
            diags.append(RED("⚠ All exits blocked by fire at end") +
                         " — either exits ignited early or agent took too long. "
                         "Check if fire spread too fast (p_spread > 0.4).")

    if len(step_log) >= (ms_final.max_steps if ms_final else 100) - 2:
        diags.append(YELLOW("⚠ Timeout") +
                     " — agent ran out of steps. "
                     "Either map is too large to navigate in 100 steps, "
                     "or looping wasted too many steps.")

    progress_steps = sum(1 for s in step_log if s["exit_delta"] > 0)
    regress_steps  = sum(1 for s in step_log if s["exit_delta"] < 0)
    if regress_steps > progress_steps:
        diags.append(YELLOW("⚠ More regression than progress") +
                     f" — agent moved away from exit more often ({regress_steps}x) "
                     f"than toward it ({progress_steps}x).")

    if not diags:
        diags.append(GREEN("No obvious single failure mode. "
                           "Run multiple seeds to find patterns."))

    for d in diags:
        print(f"  • {d}\n")

    print(BOLD(f"{'='*70}\n"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug a single Pyre PPO episode with rich diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",     type=str, default="artifacts/pyre_ppo_hard_ckpt.pt",
                   help="Path to the .pt checkpoint to load")
    p.add_argument("--difficulty",     type=str, default="hard",
                   choices=("easy", "medium", "hard"))
    p.add_argument("--server",         type=str, default="http://localhost:8000")
    p.add_argument("--stochastic",     action="store_true", default=False,
                   help="Use stochastic policy (default: deterministic greedy)")
    p.add_argument("--history-length", type=int, default=4)
    p.add_argument("--hidden-sizes",   type=str, default="512,256,128")
    p.add_argument("--seed",           type=int, default=None,
                   help="Optional fixed seed for reproducibility")
    p.add_argument("--map",            action="store_true", default=False,
                   help="Print ASCII map every 10 steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))
    debug_episode(
        checkpoint_path=args.checkpoint,
        difficulty=args.difficulty,
        server=args.server,
        deterministic=not args.stochastic,
        history_length=args.history_length,
        hidden_sizes=hidden_sizes,
        seed=args.seed,
        show_map=args.map,
    )


if __name__ == "__main__":
    main()
