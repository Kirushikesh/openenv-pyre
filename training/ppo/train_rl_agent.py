"""Train a deep RL baseline directly against the local Pyre environment.

This script makes the environment contract explicit:
  - Observation: encoded from `PyreObservation.map_state` into a fixed-length vector
  - Action: fixed discrete action table with a runtime validity mask from `available_actions_hint`
  - Reward: the environment's composite reward returned by `PyreEnvironment.step()`

It uses a self-contained NumPy actor-critic implementation so it can run in
this repository without external ML dependencies.

Examples:
    python examples/train_rl_agent.py --episodes 150 --difficulty easy
    python examples/train_rl_agent.py --episodes 300 --difficulty-schedule easy,medium
    python examples/train_rl_agent.py --episodes 200 --difficulty easy,medium,hard --observation-mode full
    python examples/train_rl_agent.py --describe-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from pyre_env.models import PyreAction, PyreObservation
from pyre_env.server.pyre_env_environment import PyreEnvironment


MAX_GRID_W = 24
MAX_GRID_H = 24
MAX_DOORS = 16
DIRECTIONS = ("north", "south", "west", "east")
WINDS = ("CALM", "NORTH", "SOUTH", "WEST", "EAST")
DIFFICULTIES = ("easy", "medium", "hard")

MOVE_KEYS = [f"move(direction='{d}')" for d in DIRECTIONS]
LOOK_KEYS = [f"look(direction='{d}')" for d in DIRECTIONS]
WAIT_KEY = "wait()"
OPEN_KEYS = [f"door(target_id='door_{i}', door_state='open')" for i in range(1, MAX_DOORS + 1)]
CLOSE_KEYS = [f"door(target_id='door_{i}', door_state='close')" for i in range(1, MAX_DOORS + 1)]
ACTION_KEYS = MOVE_KEYS + LOOK_KEYS + [WAIT_KEY] + OPEN_KEYS + CLOSE_KEYS
ACTION_DIM = len(ACTION_KEYS)
ACTION_TO_INDEX = {key: idx for idx, key in enumerate(ACTION_KEYS)}

_MOVE_RE = re.compile(r"move\(direction='(north|south|west|east)'\)")
_LOOK_RE = re.compile(r"look\(direction='(north|south|west|east)'\)")
_DOOR_RE = re.compile(r"door\(target_id='(door_(\d+))', door_state='(open|close)'\)")


def _one_hot(index: int, size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        arr[index] = 1.0
    return arr


def action_index_to_env_action(index: int) -> PyreAction:
    if 0 <= index < 4:
        return PyreAction(action="move", direction=DIRECTIONS[index])
    if 4 <= index < 8:
        return PyreAction(action="look", direction=DIRECTIONS[index - 4])
    if index == 8:
        return PyreAction(action="wait")
    if 9 <= index < 9 + MAX_DOORS:
        door_id = f"door_{index - 8}"
        return PyreAction(action="door", target_id=door_id, door_state="open")
    door_slot = index - (9 + MAX_DOORS)
    door_id = f"door_{door_slot + 1}"
    return PyreAction(action="door", target_id=door_id, door_state="close")


def build_action_mask(observation: PyreObservation) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for hint in observation.available_actions_hint:
        idx = ACTION_TO_INDEX.get(hint)
        if idx is not None:
            mask[idx] = 1.0
            continue

        match = _MOVE_RE.fullmatch(hint)
        if match:
            mask[ACTION_TO_INDEX[f"move(direction='{match.group(1)}')"]] = 1.0
            continue

        match = _LOOK_RE.fullmatch(hint)
        if match:
            mask[ACTION_TO_INDEX[f"look(direction='{match.group(1)}')"]] = 1.0
            continue

        match = _DOOR_RE.fullmatch(hint)
        if match:
            door_id = match.group(1)
            door_num = int(match.group(2))
            state = match.group(3)
            if 1 <= door_num <= MAX_DOORS:
                mask[ACTION_TO_INDEX[f"door(target_id='{door_id}', door_state='{state}')"]] = 1.0

    if mask.sum() == 0:
        mask[ACTION_TO_INDEX[WAIT_KEY]] = 1.0
    return mask


class ObservationEncoder:
    """Encode Pyre observations into a fixed-size float vector."""

    def __init__(self, mode: str = "visible"):
        if mode not in {"visible", "full"}:
            raise ValueError(f"Unsupported observation mode: {mode}")
        self.mode = mode
        self.base_dim = MAX_GRID_W * MAX_GRID_H * 10 + 22

    def encode(self, observation: PyreObservation) -> np.ndarray:
        map_state = observation.map_state
        if map_state is None:
            raise ValueError("PyreObservation.map_state is required for RL training.")

        cell_one_hot = np.zeros((MAX_GRID_H, MAX_GRID_W, 6), dtype=np.float32)
        fire_channel = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        smoke_channel = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        visible_channel = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        agent_channel = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)

        visible = {(x, y) for x, y in map_state.visible_cells}
        for y in range(map_state.grid_h):
            for x in range(map_state.grid_w):
                if self.mode == "visible" and (x, y) not in visible and (x, y) != (map_state.agent_x, map_state.agent_y):
                    continue
                i = y * map_state.grid_w + x
                cell_type = int(map_state.cell_grid[i])
                if 0 <= cell_type <= 5:
                    cell_one_hot[y, x, cell_type] = 1.0
                fire_channel[y, x] = float(map_state.fire_grid[i])
                smoke_channel[y, x] = float(map_state.smoke_grid[i])
                visible_channel[y, x] = 1.0 if (x, y) in visible else 0.0

        if 0 <= map_state.agent_x < MAX_GRID_W and 0 <= map_state.agent_y < MAX_GRID_H:
            agent_channel[map_state.agent_y, map_state.agent_x] = 1.0

        grid_features = np.concatenate(
            [
                cell_one_hot.reshape(-1),
                fire_channel.reshape(-1),
                smoke_channel.reshape(-1),
                visible_channel.reshape(-1),
                agent_channel.reshape(-1),
            ]
        )

        metadata = observation.metadata or {}
        wind_dir = str(metadata.get("wind_dir", map_state.wind_dir or "CALM")).upper()
        difficulty = str(metadata.get("difficulty", "medium")).lower()
        wind_index = WINDS.index(wind_dir) if wind_dir in WINDS else 0
        difficulty_index = DIFFICULTIES.index(difficulty) if difficulty in DIFFICULTIES else 1

        global_features = np.concatenate(
            [
                np.array(
                    [
                        float(observation.agent_health) / 100.0,
                        float(map_state.agent_health) / 100.0,
                        float(map_state.step_count) / max(1, map_state.max_steps),
                        float(map_state.fire_spread_rate),
                        float(map_state.humidity),
                        float(map_state.agent_x) / max(1, map_state.grid_w - 1),
                        float(map_state.agent_y) / max(1, map_state.grid_h - 1),
                        float(metadata.get("nearest_exit_distance", MAX_GRID_W + MAX_GRID_H) or 0.0) / float(MAX_GRID_W + MAX_GRID_H),
                        float(metadata.get("reachable_exit_count", 0.0)) / 4.0,
                        float(metadata.get("visible_cell_count", 0.0)) / float(MAX_GRID_W * MAX_GRID_H),
                        float(metadata.get("fire_sources", 0.0)) / 5.0,
                        {"none": 0.0, "light": 0.33, "moderate": 0.66, "heavy": 1.0}.get(observation.smoke_level, 0.0),
                        1.0 if map_state.agent_alive else 0.0,
                        1.0 if map_state.agent_evacuated else 0.0,
                    ],
                    dtype=np.float32,
                ),
                _one_hot(wind_index, len(WINDS)),
                _one_hot(difficulty_index, len(DIFFICULTIES)),
            ]
        )

        return np.concatenate([grid_features, global_features]).astype(np.float32)

    def describe(self, history_length: int) -> str:
        grid_text = (
            f"Observation mode `{self.mode}` encodes a {MAX_GRID_W}x{MAX_GRID_H} padded map with "
            "10 channels per cell: 6-way cell type one-hot, fire intensity, smoke intensity, visible mask, and agent mask."
        )
        if self.mode == "visible":
            visibility_text = "Only currently visible cells are populated; unseen cells stay zeroed."
        else:
            visibility_text = "The full ground-truth map is exposed for curriculum/debug use."
        return (
            f"{grid_text} {visibility_text} "
            f"Global features add health, step progress, fire parameters, position, exit-distance metadata, smoke severity, wind, and difficulty. "
            f"{history_length} encoded frames are stacked, so the network input dimension is {self.base_dim * history_length}."
        )


def softmax_with_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_logits = np.where(mask > 0.0, logits, -1e9)
    max_logits = np.max(masked_logits, axis=1, keepdims=True)
    exps = np.exp(masked_logits - max_logits) * mask
    denom = np.sum(exps, axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exps / denom


class AdamOptimizer:
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], clip_norm: float = 1.0) -> None:
        total_norm_sq = 0.0
        for grad in grads.values():
            total_norm_sq += float(np.sum(grad * grad))
        total_norm = math.sqrt(total_norm_sq)
        scale = 1.0
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)

        self.t += 1
        for name, param in params.items():
            grad = grads[name] * scale
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class PolicyValueNetwork:
    def __init__(self, input_dim: int, action_dim: int, rng: np.random.Generator, hidden_sizes: Sequence[int] = (256, 128)):
        h1, h2 = hidden_sizes
        self.params: Dict[str, np.ndarray] = {
            "w1": self._init_weight(rng, input_dim, h1),
            "b1": np.zeros(h1, dtype=np.float32),
            "w2": self._init_weight(rng, h1, h2),
            "b2": np.zeros(h2, dtype=np.float32),
            "wp": self._init_weight(rng, h2, action_dim),
            "bp": np.zeros(action_dim, dtype=np.float32),
            "wv": self._init_weight(rng, h2, 1),
            "bv": np.zeros(1, dtype=np.float32),
        }
        self.optimizer = AdamOptimizer(self.params)

    @staticmethod
    def _init_weight(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
        scale = math.sqrt(2.0 / max(1, in_dim + out_dim))
        return (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        z1 = x @ self.params["w1"] + self.params["b1"]
        h1 = np.tanh(z1)
        z2 = h1 @ self.params["w2"] + self.params["b2"]
        h2 = np.tanh(z2)
        logits = h2 @ self.params["wp"] + self.params["bp"]
        values = (h2 @ self.params["wv"] + self.params["bv"]).reshape(-1)
        cache = {"x": x, "h1": h1, "h2": h2}
        return logits, values, cache

    def predict(self, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        logits, values, _ = self.forward(x[None, :])
        probs = softmax_with_mask(logits, mask[None, :])[0]
        return probs, float(values[0])

    def update(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        value_coef: float = 0.5,
    ) -> Dict[str, float]:
        logits, values, cache = self.forward(states)
        probs = softmax_with_mask(logits, masks)

        batch_size = max(1, states.shape[0])
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), actions] -= 1.0
        grad_logits *= advantages[:, None] / batch_size
        grad_logits *= masks

        grad_values = ((values - returns)[:, None] * value_coef) / batch_size

        grads: Dict[str, np.ndarray] = {}
        grads["wp"] = cache["h2"].T @ grad_logits
        grads["bp"] = np.sum(grad_logits, axis=0)
        grads["wv"] = cache["h2"].T @ grad_values
        grads["bv"] = np.sum(grad_values, axis=0)

        dh2 = grad_logits @ self.params["wp"].T + grad_values @ self.params["wv"].T
        dz2 = dh2 * (1.0 - cache["h2"] ** 2)
        grads["w2"] = cache["h1"].T @ dz2
        grads["b2"] = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.params["w2"].T
        dz1 = dh1 * (1.0 - cache["h1"] ** 2)
        grads["w1"] = cache["x"].T @ dz1
        grads["b1"] = np.sum(dz1, axis=0)

        self.optimizer.step(self.params, grads, clip_norm=1.0)

        chosen_probs = np.clip(probs[np.arange(batch_size), actions], 1e-8, 1.0)
        policy_loss = float(-np.mean(advantages * np.log(chosen_probs)))
        value_loss = float(0.5 * np.mean((values - returns) ** 2))
        entropy = float(-np.mean(np.sum(np.where(probs > 0.0, probs * np.log(np.clip(probs, 1e-8, 1.0)), 0.0), axis=1)))
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "mean_value": float(np.mean(values)),
        }

    def save(self, path: Path, metadata: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: value for name, value in self.params.items()}
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez(path, **arrays)


@dataclass
class Trajectory:
    states: List[np.ndarray]
    masks: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    evacuated: bool
    final_health: float
    steps: int
    total_reward: float


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    advantages = np.zeros(len(rewards_arr), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for i in range(len(rewards_arr) - 1, -1, -1):
        delta = rewards_arr[i] + gamma * next_value - values_arr[i]
        gae = delta + gamma * gae_lambda * gae
        advantages[i] = gae
        next_value = values_arr[i]
    returns = advantages + values_arr
    return returns.astype(np.float32), advantages.astype(np.float32)


def select_action(
    network: PolicyValueNetwork,
    state_vec: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    greedy: bool = False,
) -> tuple[int, float]:
    probs, value = network.predict(state_vec, mask)
    valid_indices = np.flatnonzero(mask > 0.0)
    if len(valid_indices) == 0:
        return ACTION_TO_INDEX[WAIT_KEY], value
    if greedy:
        best_local = int(np.argmax(probs[valid_indices]))
        return int(valid_indices[best_local]), value
    return int(rng.choice(np.arange(len(probs)), p=probs)), value


def build_stacked_state(frames: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(frames), dtype=np.float32)


def run_episode(
    env: PyreEnvironment,
    network: PolicyValueNetwork,
    encoder: ObservationEncoder,
    rng: np.random.Generator,
    difficulty: str,
    history_length: int,
    greedy: bool = False,
) -> Trajectory:
    observation = env.reset(difficulty=difficulty)
    zero_frame = np.zeros(encoder.base_dim, dtype=np.float32)
    frames: deque[np.ndarray] = deque([zero_frame.copy() for _ in range(history_length)], maxlen=history_length)
    frames.append(encoder.encode(observation))

    states: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    values: List[float] = []

    total_reward = 0.0
    final_health = observation.agent_health
    evacuated = False
    steps = 0

    while True:
        state_vec = build_stacked_state(frames)
        mask = build_action_mask(observation)
        action_idx, value = select_action(network, state_vec, mask, rng, greedy=greedy)
        action = action_index_to_env_action(action_idx)

        next_obs = env.step(action)
        reward = float(next_obs.reward or 0.0)

        states.append(state_vec)
        masks.append(mask)
        actions.append(action_idx)
        rewards.append(reward)
        values.append(value)

        total_reward += reward
        steps += 1
        final_health = next_obs.agent_health
        evacuated = next_obs.agent_evacuated

        frames.append(encoder.encode(next_obs))
        observation = next_obs
        if next_obs.done:
            break

    return Trajectory(
        states=states,
        masks=masks,
        actions=actions,
        rewards=rewards,
        values=values,
        evacuated=evacuated,
        final_health=final_health,
        steps=steps,
        total_reward=total_reward,
    )


def evaluate_policy(
    env: PyreEnvironment,
    network: PolicyValueNetwork,
    encoder: ObservationEncoder,
    rng: np.random.Generator,
    difficulty: str,
    history_length: int,
    episodes: int,
) -> Dict[str, float]:
    rewards = []
    evacuations = 0
    lengths = []
    for _ in range(episodes):
        traj = run_episode(env, network, encoder, rng, difficulty, history_length, greedy=True)
        rewards.append(traj.total_reward)
        lengths.append(traj.steps)
        evacuations += int(traj.evacuated)
    return {
        "eval_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "eval_reward_max": float(np.max(rewards)) if rewards else 0.0,
        "eval_success_rate": float(evacuations / max(1, episodes)),
        "eval_steps_mean": float(np.mean(lengths)) if lengths else 0.0,
    }


def expand_difficulty_schedule(schedule_text: str, episodes: int) -> List[str]:
    stages = [part.strip().lower() for part in schedule_text.split(",") if part.strip()]
    if not stages:
        stages = ["medium"]
    for stage in stages:
        if stage not in DIFFICULTIES:
            raise ValueError(f"Invalid difficulty in schedule: {stage}")
    segment = max(1, episodes // len(stages))
    expanded: List[str] = []
    for stage in stages:
        expanded.extend([stage] * segment)
    while len(expanded) < episodes:
        expanded.append(stages[-1])
    return expanded[:episodes]


def describe_environment_contract(encoder: ObservationEncoder, history_length: int) -> str:
    action_text = (
        f"Action space has {ACTION_DIM} fixed discrete actions: 4 moves, 4 looks, wait, "
        f"{MAX_DOORS} door-open slots, and {MAX_DOORS} door-close slots. "
        "A per-step mask from `available_actions_hint` prevents invalid actions."
    )
    reward_text = (
        "Reward comes directly from the environment's composite rubric: time penalty, exit progress, "
        "progress regression penalty, safe-progress bonus, danger penalty, health-drain penalty, "
        "strategic door bonus, exploration bonus, plus terminal evacuation/death/timeout/near-miss/time bonuses."
    )
    return "\n".join(
        [
            "Pyre RL contract",
            encoder.describe(history_length),
            action_text,
            reward_text,
        ]
    )


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    run = 0.0
    q: deque[float] = deque()
    for value in values:
        q.append(float(value))
        run += float(value)
        if len(q) > window:
            run -= q.popleft()
        out.append(run / len(q))
    return out


def save_metrics_csv(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_training_graph(path: Path, episode_rows: List[Dict[str, float | int | str]], eval_rows: List[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not episode_rows:
        return

    width = 1260
    height = 780
    margin_left = 100   # extra room for rotated Y-axis label + tick values
    margin_right = 110  # extra room for right axis label + tick values
    margin_top = 70     # room for title
    margin_bottom = 90  # room for X-axis label + tick values + legend
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # X: plot_left=100, plot_right=1150  Y: plot_top=70, plot_bottom=690

    episodes = [int(r["episode"]) for r in episode_rows]
    rewards = [float(r["reward"]) for r in episode_rows]
    reward_ma = _moving_average(rewards, 20)
    success_ma = _moving_average([float(r["evacuated"]) for r in episode_rows], 20)

    all_reward_values = rewards + reward_ma + [float(r["reward_mean"]) for r in eval_rows] + [float(r["reward_max"]) for r in eval_rows]
    y_min = min(all_reward_values) if all_reward_values else -1.0
    y_max = max(all_reward_values) if all_reward_values else 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    y_pad = 0.1 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    max_episode = max(episodes) if episodes else 1

    plot_left = margin_left
    plot_right = margin_left + plot_w
    plot_top = margin_top
    plot_bottom = margin_top + plot_h

    def x_pos(ep: float) -> float:
        return plot_left + (float(ep) - 1.0) / max(1.0, max_episode - 1.0) * plot_w

    def y_pos_reward(value: float) -> float:
        return plot_top + (y_max - float(value)) / max(1e-6, (y_max - y_min)) * plot_h

    def y_pos_success(value: float) -> float:
        return plot_top + (1.0 - float(value)) * plot_h

    def polyline(points: List[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    reward_points    = [(x_pos(ep), y_pos_reward(val)) for ep, val in zip(episodes, rewards)]
    reward_ma_points = [(x_pos(ep), y_pos_reward(val)) for ep, val in zip(episodes, reward_ma)]
    success_points   = [(x_pos(ep), y_pos_success(val)) for ep, val in zip(episodes, success_ma)]
    eval_points      = [(x_pos(float(r["episode"])), y_pos_success(float(r["success_rate"]))) for r in eval_rows]

    n_x_ticks = 8
    episode_ticks = sorted(set(
        max(1, round(1 + i * (max_episode - 1) / n_x_ticks))
        for i in range(n_x_ticks + 1)
    ))
    n_y_ticks = 6
    reward_ticks  = [y_min + (y_max - y_min) * i / n_y_ticks for i in range(n_y_ticks + 1)]
    success_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')

    # Background
    svg.append('<rect width="100%" height="100%" fill="#f7f5ef"/>')

    # Title + subtitle
    svg.append(f'<text x="{plot_left}" y="28" font-family="Georgia, serif" font-size="22" font-weight="bold" fill="#1d2a38">Pyre RL Training</text>')
    svg.append(f'<text x="{plot_left}" y="50" font-family="Georgia, serif" font-size="13" fill="#5b6770">Left axis: Reward   |   Right axis: Success Rate (0–1)</text>')

    # Plot area background + border
    svg.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#fffdf8" stroke="#b8b0a2" stroke-width="1.5"/>')

    # ── Vertical grid lines + X-axis ticks ──────────────────────────────────
    for tick in episode_ticks:
        x = x_pos(float(tick))
        # dashed grid line
        svg.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" '
                   f'stroke="#d8d2c8" stroke-width="1" stroke-dasharray="4,4"/>')
        # solid tick mark on bottom axis
        svg.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 6}" '
                   f'stroke="#6b6460" stroke-width="1.5"/>')
        # tick label
        svg.append(f'<text x="{x:.1f}" y="{plot_bottom + 20}" text-anchor="middle" '
                   f'font-family="Georgia, serif" font-size="12" fill="#4a4540">{tick}</text>')

    # X-axis title
    x_title_x = plot_left + plot_w / 2
    x_title_y = plot_bottom + 50
    svg.append(f'<text x="{x_title_x:.1f}" y="{x_title_y}" text-anchor="middle" '
               f'font-family="Georgia, serif" font-size="14" font-weight="bold" fill="#1d2a38">Episode</text>')

    # ── Horizontal grid lines + Left Y-axis ticks (Reward) ──────────────────
    for tick in reward_ticks:
        y = y_pos_reward(tick)
        # dashed grid line
        svg.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
                   f'stroke="#d8d2c8" stroke-width="1" stroke-dasharray="4,4"/>')
        # solid tick mark on left axis
        svg.append(f'<line x1="{plot_left - 6}" y1="{y:.1f}" x2="{plot_left}" y2="{y:.1f}" '
                   f'stroke="#6b6460" stroke-width="1.5"/>')
        # tick label
        svg.append(f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-family="Georgia, serif" font-size="12" fill="#8a4b08">{tick:.1f}</text>')

    # Left Y-axis title (rotated) — centered on plot height
    ly_cx = plot_left - 70
    ly_cy = plot_top + plot_h / 2
    svg.append(f'<text transform="rotate(-90, {ly_cx:.1f}, {ly_cy:.1f})" '
               f'x="{ly_cx:.1f}" y="{ly_cy:.1f}" text-anchor="middle" '
               f'font-family="Georgia, serif" font-size="14" font-weight="bold" fill="#8a4b08">Reward</text>')

    # ── Right Y-axis ticks (Success Rate) ───────────────────────────────────
    for tick in success_ticks:
        y = y_pos_success(tick)
        # solid tick mark on right axis
        svg.append(f'<line x1="{plot_right}" y1="{y:.1f}" x2="{plot_right + 6}" y2="{y:.1f}" '
                   f'stroke="#6b6460" stroke-width="1.5"/>')
        # tick label
        svg.append(f'<text x="{plot_right + 12}" y="{y + 4:.1f}" '
                   f'font-family="Georgia, serif" font-size="12" fill="#0d5b6b">{tick:.2f}</text>')

    # Right Y-axis title (rotated)
    ry_cx = plot_right + 85
    ry_cy = plot_top + plot_h / 2
    svg.append(f'<text transform="rotate(90, {ry_cx:.1f}, {ry_cy:.1f})" '
               f'x="{ry_cx:.1f}" y="{ry_cy:.1f}" text-anchor="middle" '
               f'font-family="Georgia, serif" font-size="14" font-weight="bold" fill="#0d5b6b">Success Rate</text>')

    # ── Axis border lines (solid, on top of grid) ────────────────────────────
    # Bottom axis
    svg.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" '
               f'stroke="#6b6460" stroke-width="2"/>')
    # Left axis
    svg.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" '
               f'stroke="#6b6460" stroke-width="2"/>')
    # Right axis
    svg.append(f'<line x1="{plot_right}" y1="{plot_top}" x2="{plot_right}" y2="{plot_bottom}" '
               f'stroke="#6b6460" stroke-width="2"/>')

    # ── Data series ─────────────────────────────────────────────────────────
    # Raw episode reward (faint)
    svg.append(f'<polyline fill="none" stroke="#c5bfb1" stroke-width="1.5" points="{polyline(reward_points)}"/>')
    # Reward moving average
    svg.append(f'<polyline fill="none" stroke="#c1661c" stroke-width="3" stroke-linejoin="round" points="{polyline(reward_ma_points)}"/>')
    # Success moving average
    svg.append(f'<polyline fill="none" stroke="#127a8a" stroke-width="3" stroke-linejoin="round" points="{polyline(success_points)}"/>')
    # Eval checkpoints
    for x, y in eval_points:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#0d5b6b" stroke="#ffffff" stroke-width="2"/>')

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_y = plot_bottom + 72
    items = [
        ("#c1661c", 3,   False, "Reward (moving avg)"),
        ("#127a8a", 3,   False, "Success rate (moving avg)"),
        ("#c5bfb1", 1.5, False, "Episode reward"),
        ("#0d5b6b", 0,   True,  "Eval success checkpoint"),
    ]
    lx = plot_left
    for color, sw, is_dot, label in items:
        if is_dot:
            svg.append(f'<circle cx="{lx + 15}" cy="{legend_y - 4}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        else:
            svg.append(f'<line x1="{lx}" y1="{legend_y - 4}" x2="{lx + 30}" y2="{legend_y - 4}" stroke="{color}" stroke-width="{sw}"/>')
        svg.append(f'<text x="{lx + 36}" y="{legend_y}" font-family="Georgia, serif" font-size="12" fill="#1d2a38">{label}</text>')
        lx += 230

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    encoder = ObservationEncoder(mode=args.observation_mode)
    difficulty_schedule = expand_difficulty_schedule(args.difficulty_schedule, args.episodes)
    input_dim = encoder.base_dim * args.history_length
    network = PolicyValueNetwork(input_dim=input_dim, action_dim=ACTION_DIM, rng=rng)
    env = PyreEnvironment(max_steps=args.max_steps)

    print(describe_environment_contract(encoder, args.history_length))
    print("")

    batch_states: List[np.ndarray] = []
    batch_masks: List[np.ndarray] = []
    batch_actions: List[int] = []
    batch_returns: List[np.ndarray] = []
    batch_advantages: List[np.ndarray] = []

    reward_window: deque[float] = deque(maxlen=20)
    success_window: deque[float] = deque(maxlen=20)
    episode_metrics: List[Dict[str, float | int | str]] = []
    eval_metrics_rows: List[Dict[str, float | int | str]] = []

    for episode_idx in range(args.episodes):
        difficulty = difficulty_schedule[episode_idx] if args.difficulty_schedule else args.difficulty
        traj = run_episode(
            env=env,
            network=network,
            encoder=encoder,
            rng=rng,
            difficulty=difficulty,
            history_length=args.history_length,
            greedy=False,
        )

        returns, advantages = compute_gae(traj.rewards, traj.values, args.gamma, args.gae_lambda)
        batch_states.extend(traj.states)
        batch_masks.extend(traj.masks)
        batch_actions.extend(traj.actions)
        batch_returns.append(returns)
        batch_advantages.append(advantages)

        reward_window.append(traj.total_reward)
        success_window.append(float(traj.evacuated))
        episode_metrics.append(
            {
                "episode": episode_idx + 1,
                "difficulty": difficulty,
                "reward": round(traj.total_reward, 4),
                "evacuated": int(traj.evacuated),
                "steps": traj.steps,
                "final_health": round(traj.final_health, 2),
                "reward_mean_20": round(float(np.mean(reward_window)), 4),
                "success_rate_20": round(float(np.mean(success_window)), 4),
            }
        )

        print(
            f"episode={episode_idx + 1:04d} difficulty={difficulty:<6} "
            f"steps={traj.steps:03d} reward={traj.total_reward:+8.3f} "
            f"evacuated={int(traj.evacuated)} health={traj.final_health:6.1f}"
        )

        should_update = (episode_idx + 1) % args.update_every == 0 or (episode_idx + 1) == args.episodes
        if should_update and batch_states:
            states_arr = np.asarray(batch_states, dtype=np.float32)
            masks_arr = np.asarray(batch_masks, dtype=np.float32)
            actions_arr = np.asarray(batch_actions, dtype=np.int64)
            returns_arr = np.concatenate(batch_returns).astype(np.float32)
            advantages_arr = np.concatenate(batch_advantages).astype(np.float32)
            advantages_arr = (advantages_arr - advantages_arr.mean()) / (advantages_arr.std() + 1e-8)

            network.optimizer.lr = args.learning_rate
            metrics = {}
            for _ in range(args.update_epochs):
                order = rng.permutation(len(states_arr))
                for start in range(0, len(states_arr), args.minibatch_size):
                    idx = order[start:start + args.minibatch_size]
                    metrics = network.update(
                        states=states_arr[idx],
                        masks=masks_arr[idx],
                        actions=actions_arr[idx],
                        returns=returns_arr[idx],
                        advantages=advantages_arr[idx],
                        value_coef=args.value_coef,
                    )

            print(
                f"update  episodes={episode_idx + 1:04d} samples={len(states_arr):05d} "
                f"reward_mean20={np.mean(reward_window):+8.3f} success20={np.mean(success_window):.2f} "
                f"policy_loss={metrics['policy_loss']:+.4f} value_loss={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}"
            )

            batch_states.clear()
            batch_masks.clear()
            batch_actions.clear()
            batch_returns.clear()
            batch_advantages.clear()

        should_eval = args.eval_every > 0 and ((episode_idx + 1) % args.eval_every == 0 or (episode_idx + 1) == args.episodes)
        if should_eval:
            eval_metrics = evaluate_policy(
                env=env,
                network=network,
                encoder=encoder,
                rng=rng,
                difficulty=args.eval_difficulty,
                history_length=args.history_length,
                episodes=args.eval_episodes,
            )
            print(
                f"eval    episodes={episode_idx + 1:04d} difficulty={args.eval_difficulty:<6} "
                f"reward_mean={eval_metrics['eval_reward_mean']:+8.3f} "
                f"reward_max={eval_metrics['eval_reward_max']:+8.3f} "
                f"success={eval_metrics['eval_success_rate']:.2f} "
                f"steps={eval_metrics['eval_steps_mean']:.1f}"
            )
            eval_metrics_rows.append(
                {
                    "episode": episode_idx + 1,
                    "difficulty": args.eval_difficulty,
                    "reward_mean": round(eval_metrics["eval_reward_mean"], 4),
                    "reward_max": round(eval_metrics["eval_reward_max"], 4),
                    "success_rate": round(eval_metrics["eval_success_rate"], 4),
                    "steps_mean": round(eval_metrics["eval_steps_mean"], 4),
                }
            )

    if args.output:
        output_path = Path(args.output)
        network.save(
            output_path,
            metadata={
                "observation_mode": args.observation_mode,
                "history_length": args.history_length,
                "episodes": args.episodes,
                "difficulty": args.difficulty,
                "difficulty_schedule": args.difficulty_schedule,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "learning_rate": args.learning_rate,
                "update_epochs": args.update_epochs,
                "minibatch_size": args.minibatch_size,
                "action_dim": ACTION_DIM,
                "input_dim": input_dim,
            },
        )
        print(f"saved   model={output_path}")
        if args.save_metrics:
            metrics_path = output_path.with_suffix(".csv")
            save_metrics_csv(metrics_path, episode_metrics)
            print(f"saved   metrics={metrics_path}")
        if args.save_graph:
            graph_path = output_path.with_suffix(".svg")
            save_training_graph(graph_path, episode_metrics, eval_metrics_rows)
            print(f"saved   graph={graph_path}")
            # Also save PNG
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.ticker as mticker
                import matplotlib.patches as mpatches

                episodes_list  = [int(r["episode"])    for r in episode_metrics]
                rewards_list   = [float(r["reward"])   for r in episode_metrics]
                evacuated_list = [float(r["evacuated"]) for r in episode_metrics]
                diff_list      = [str(r["difficulty"]) for r in episode_metrics]

                def _ma(vals, w=20):
                    out, run, q = [], 0.0, []
                    for v in vals:
                        q.append(v); run += v
                        if len(q) > w: run -= q.pop(0)
                        out.append(run / len(q))
                    return out

                reward_ma  = _ma(rewards_list)
                success_ma = _ma(evacuated_list)
                eval_eps   = [int(r["episode"])       for r in eval_metrics_rows]
                eval_succ  = [float(r["success_rate"]) for r in eval_metrics_rows]

                diff_colors = {"easy": "#d4edda", "medium": "#fff3cd", "hard": "#f8d7da"}
                regions = []
                if diff_list:
                    cur, start = diff_list[0], episodes_list[0]
                    for ep, d in zip(episodes_list[1:], diff_list[1:]):
                        if d != cur:
                            regions.append((start, ep, cur)); cur, start = d, ep
                    regions.append((start, episodes_list[-1], cur))

                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()
                for x0, x1, diff in regions:
                    ax1.axvspan(x0, x1, color=diff_colors.get(diff, "#eeeeee"), alpha=0.35, zorder=0)
                ax1.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
                ax1.plot(episodes_list, rewards_list,  color="#d1c7bc", linewidth=0.8, alpha=0.6, label="Episode reward", zorder=2)
                ax1.plot(episodes_list, reward_ma,     color="#c1661c", linewidth=2.5, label="Reward (MA-20)", zorder=3)
                ax2.plot(episodes_list, success_ma,    color="#1a7a8a", linewidth=2.5, label="Success rate (MA-20)", zorder=3)
                if eval_eps:
                    ax2.scatter(eval_eps, eval_succ, color="#0d5b6b", s=60, zorder=5, marker="D", edgecolors="white", linewidths=1.2, label="Eval success")
                ax1.set_xlabel("Episode", fontsize=13, fontweight="bold", labelpad=8)
                ax1.set_ylabel("Reward",  fontsize=13, fontweight="bold", color="#c1661c", labelpad=8)
                ax2.set_ylabel("Success Rate", fontsize=13, fontweight="bold", color="#1a7a8a", labelpad=8)
                ax1.tick_params(axis="y", labelcolor="#c1661c")
                ax2.tick_params(axis="y", labelcolor="#1a7a8a")
                ax2.set_ylim(-0.05, 1.05)
                ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
                ax1.grid(True, linestyle="--", linewidth=0.6, color="#dddddd", alpha=0.8)
                ax1.set_xlim(episodes_list[0], episodes_list[-1])
                diff_patches = [mpatches.Patch(color=diff_colors[d], alpha=0.6, label=d.capitalize())
                                for d in ["easy", "medium", "hard"] if d in diff_list]
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2 + diff_patches, l1 + l2 + [p.get_label() for p in diff_patches],
                           loc="upper left", fontsize=9, framealpha=0.85)
                final_sr = success_ma[-1] if success_ma else 0.0
                fig.suptitle(f"Pyre NumPy A2C Training  —  {episodes_list[-1]} episodes  |  final success: {final_sr:.0%}",
                             fontsize=14, fontweight="bold", y=1.01)
                fig.tight_layout()
                png_path = output_path.with_suffix(".png")
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"saved   graph_png={png_path}")
            except ImportError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy actor-critic baseline for Pyre.")
    parser.add_argument("--episodes", type=int, default=120, help="Training episodes.")
    parser.add_argument("--difficulty", type=str, default="easy", choices=DIFFICULTIES)
    parser.add_argument(
        "--difficulty-schedule",
        type=str,
        default="easy,medium",
        help="Comma-separated curriculum, expanded evenly across episodes.",
    )
    parser.add_argument("--eval-difficulty", type=str, default="medium", choices=DIFFICULTIES)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--update-every", type=int, default=5, help="Episodes per policy update.")
    parser.add_argument("--update-epochs", type=int, default=3, help="Gradient passes over each on-policy batch.")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Samples per gradient step.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--observation-mode", type=str, default="visible", choices=("visible", "full"))
    parser.add_argument("--output", type=str, default="artifacts/pyre_actor_critic.npz")
    parser.add_argument("--save-metrics", action="store_true", help="Save per-episode metrics as CSV beside the model.")
    parser.add_argument("--save-graph", action="store_true", help="Save an SVG training graph beside the model.")
    parser.add_argument("--describe-only", action="store_true", help="Print observation/action/reward definitions and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoder = ObservationEncoder(mode=args.observation_mode)
    if args.describe_only:
        print(describe_environment_contract(encoder, args.history_length))
        return
    train(args)


if __name__ == "__main__":
    main()
