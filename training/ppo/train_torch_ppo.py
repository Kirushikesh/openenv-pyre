"""
PyTorch PPO Agent for Pyre — Fire Evacuation RL Training Script.

=== ENVIRONMENT SUMMARY ===
Pyre is a partial-observability crisis navigation environment:
  - Grid: 16×16 (easy/medium) or 20×24 (hard, procedural)
  - Agent: Spawns inside a burning building, must evacuate before dying
  - Fire: Spreads via cellular automaton — wind, humidity, fuel vary per episode
  - Partial observability: visibility radius (2–5 cells) shrinks in heavy smoke
  - Doors: Can be opened/closed to slow fire spread (+0.5 strategic door bonus)
  - Health: 100 HP, drains from smoke (0.5–5/step) and fire (10/step)

=== ACTION SPACE (37 discrete; no `look` — grid encoder carries visibility) ===
  0–3   : move(north|south|west|east)
  4     : wait()
  5–20  : door(door_1..16, open)
  21–36 : door(door_1..16, close)
  Runtime action masking via `available_actions_hint` prevents invalid moves.

=== OBSERVATION ENCODING ===
  Per-step grid: 24×24 padded map × 10 channels
    • 6 one-hot cell type (floor/wall/door_open/door_closed/exit/obstacle)
    • fire intensity [0, 1]
    • smoke density  [0, 1]
    • visibility mask (1=visible, 0=unseen)
    • agent position mask
  Global scalars (17) + exit compass (3) + one-hots: wind (5) + difficulty (4) + route hint (4)
  Frame stacking: 4 consecutive frames → input_dim = 5790 × 4 = 23160

=== REWARD STRUCTURE (environment: server/rubrics.py) ===
  Per-step:
    -0.01  time penalty
    +0.25  BFS progress toward nearest unblocked exit (move only)
    -0.15  BFS regression (move only)
    +0.05  safe-progress bonus (move, progress, new cell smoke < 0.5)
    -0.50  danger (move into smoke ≥ 0.5 or cell adjacent to fire ≥ 0.3)
    -0.02×dmg  health drain (dmg = HP lost this step)
    +0.50  strategic door close (door tile cardinally adjacent to fire ≥ 0.3; once per door / ep)
    +0.02  exploration (move to first-time cell)
  Terminal:
    +5.00  evacuation success
    +1.50×(hp/100)  health survival on evac
    -10.0  death
    -5.0  timeout if no reachable exit (all blocked), else −5 − 3×(hp/100)
    0→+3.0  near-miss on death: max(0, 3 − 0.5×min BFS exit dist reached)
    +0.05×remaining_steps  time bonus on evac

  Extra shaping (this file only, applied to returned step reward before PPO):
    -0.05  if action is wait()
    -0.15  if move lands in a cell with a cardinal neighbor fire > 0.15
    -0.20  if move revisits a position in the last 12 cells
    +max(0, 0.25 − 0.04×d)  on move if not evacuated: d = Manhattan distance
                            to nearest exit cell (from map_state.exit_positions)

=== ALGORITHM: PPO (Proximal Policy Optimization) ===
WHY PPO over alternatives:
  • DQN    — Off-policy, harder credit assignment for sparse terminal rewards; no clean action masking
  • A2C    — Simpler but no clipping → unstable on hard stochastic episodes
  • SAC    — Designed for continuous spaces; discrete SAC works but adds complexity
  • LSTM-PPO — Better for fully text-only obs; grid map_state already encodes spatial state
  → PPO + frame-stack + action-mask hits the sweet spot for this env

Key PPO improvements over the existing NumPy A2C (train_rl_agent.py):
  ✓ PPO clip (ε=0.2)        prevents catastrophic updates
  ✓ Entropy regularization  sustains exploration in smoke-obscured corridors
  ✓ Value function clipping  stabilises critic under sparse terminal rewards
  ✓ GPU acceleration         10–20× faster than NumPy baseline
  ✓ LayerNorm in network     improves gradient flow for large input dims
  ✓ Linear LR decay          stabilises late-stage convergence
  ✓ Better curriculum        3-stage easy→medium→hard with patience gating

Usage:
    python training/ppo/train_torch_ppo.py --episodes 500 --device cuda
    python training/ppo/train_torch_ppo.py --episodes 300 --difficulty-schedule easy,medium,hard_fixed,hard
    python training/ppo/train_torch_ppo.py --resume artifacts/pyre_ppo_checkpoint.pt
    python training/ppo/train_torch_ppo.py --describe-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
# Optional torch import — fail fast with a helpful message
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LinearLR
except ImportError:
    sys.exit(
        "PyTorch not found. Install with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "or for CPU only:\n"
        "  pip install torch"
    )

# ---------------------------------------------------------------------------
# Project imports — support both package install and direct run from root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import requests
except ImportError:
    requests = None

try:
    from pyre_env.models import PyreAction, PyreMapState, PyreObservation
    from pyre_env.server.pyre_env_environment import PyreEnvironment
except ModuleNotFoundError:
    try:
        from models import PyreAction, PyreMapState, PyreObservation
        from server.pyre_env_environment import PyreEnvironment
    except ModuleNotFoundError:
        # If we can't import the environment, we might still be able to run in HTTP mode
        PyreEnvironment = None
        try:
            from models import PyreAction, PyreMapState, PyreObservation
        except ImportError:
            sys.exit(
                "Cannot import Pyre modules. Run this script from the openenv-pyre root:\n"
                "  python training/ppo/train_torch_ppo.py"
            )


# ---------------------------------------------------------------------------
# HTTP environment wrapper
# ---------------------------------------------------------------------------

class HttpPyreEnv:
    """Thin wrapper around the Pyre REST API.

    Exposes the same ``reset()`` / ``step()`` interface as ``PyreEnvironment``
    so the episode runner needs no changes.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 15):
        if requests is None:
            raise ImportError("HTTP mode requires 'requests' package. Install with: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _parse(self, data: Dict[str, Any]) -> PyreObservation:
        """Convert a raw JSON response dict into a PyreObservation."""
        obs_raw = data.get("observation", data)
        ms_raw = obs_raw.get("map_state")
        map_state = PyreMapState(**ms_raw) if ms_raw else None

        # Ensure all fields are correctly typed
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

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Perform an HTTP request with exponential backoff for rate limits and 5xx errors."""
        max_retries = 10
        base_delay = 1.0
        for i in range(max_retries):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code == 429:
                    # Rate limited: wait and retry
                    wait = base_delay * (2 ** i) + np.random.uniform(0, 1)
                    print(f"  [http] Rate limited (429). Retrying in {wait:.1f}s... (attempt {i+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if 500 <= resp.status_code < 600:
                    # Server error: wait and retry
                    wait = base_delay * (2 ** i)
                    print(f"  [http] Server error ({resp.status_code}). Retrying in {wait:.1f}s... (attempt {i+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if i == max_retries - 1:
                    raise
                wait = base_delay * (2 ** i)
                print(f"  [http] Network error: {e}. Retrying in {wait:.1f}s... (attempt {i+1}/{max_retries})")
                time.sleep(wait)
        raise requests.exceptions.RetryError("Max retries exceeded")

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> PyreObservation:
        payload: Dict[str, Any] = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        resp = self._request_with_retry("POST", f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        return self._parse(resp.json())

    def step(self, action: PyreAction) -> PyreObservation:
        payload: Dict[str, Any] = {"action": action.action}
        if action.direction: payload["direction"] = action.direction
        if action.target_id: payload["target_id"] = action.target_id
        if action.door_state: payload["door_state"] = action.door_state
        resp = self._request_with_retry("POST", f"{self.base_url}/step", json=payload, timeout=self.timeout)
        return self._parse(resp.json())

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/state", timeout=5)
            return r.status_code < 500
        except:
            return False


# ---------------------------------------------------------------------------
# Reuse the established observation/action interface from train_rl_agent.py
# These are the canonical definitions for this environment.
# ---------------------------------------------------------------------------
MAX_GRID_W = 24
MAX_GRID_H = 24
MAX_DOORS = 16
DIRECTIONS = ("north", "south", "west", "east")
WINDS = ("CALM", "NORTH", "SOUTH", "WEST", "EAST")
DIFFICULTIES = ("easy", "medium", "hard_fixed", "hard")

MOVE_KEYS = [f"move(direction='{d}')" for d in DIRECTIONS]
WAIT_KEY = "wait()"
OPEN_KEYS = [f"door(target_id='door_{i}', door_state='open')" for i in range(1, MAX_DOORS + 1)]
CLOSE_KEYS = [f"door(target_id='door_{i}', door_state='close')" for i in range(1, MAX_DOORS + 1)]
ACTION_KEYS = MOVE_KEYS + [WAIT_KEY] + OPEN_KEYS + CLOSE_KEYS
ACTION_DIM = len(ACTION_KEYS)  # 37
ACTION_TO_INDEX = {key: idx for idx, key in enumerate(ACTION_KEYS)}

import re
_MOVE_RE = re.compile(r"move\(direction='(north|south|west|east)'\)")
_DOOR_RE = re.compile(r"door\(target_id='(door_(\d+))', door_state='(open|close)'\)")


def action_index_to_env_action(index: int) -> PyreAction:
    if 0 <= index < 4:
        return PyreAction(action="move", direction=DIRECTIONS[index])
    if index == 4:
        return PyreAction(action="wait")
    if 5 <= index < 5 + MAX_DOORS:
        door_id = f"door_{index - 4}"
        return PyreAction(action="door", target_id=door_id, door_state="open")
    door_slot = index - (5 + MAX_DOORS)
    door_id = f"door_{door_slot + 1}"
    return PyreAction(action="door", target_id=door_id, door_state="close")


def build_action_mask(observation: PyreObservation) -> np.ndarray:
    """Build a binary validity mask over the 37-action PPO action space."""
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    for hint in observation.available_actions_hint:
        idx = ACTION_TO_INDEX.get(hint)
        if idx is not None:
            mask[idx] = 1.0
            continue
        m = _MOVE_RE.fullmatch(hint)
        if m:
            mask[ACTION_TO_INDEX[f"move(direction='{m.group(1)}')"]] = 1.0
            continue
        m = _DOOR_RE.fullmatch(hint)
        if m:
            door_id, door_num, state = m.group(1), int(m.group(2)), m.group(3)
            if 1 <= door_num <= MAX_DOORS:
                mask[ACTION_TO_INDEX[f"door(target_id='{door_id}', door_state='{state}')"]] = 1.0
    if mask.sum() == 0:
        mask[ACTION_TO_INDEX[WAIT_KEY]] = 1.0
    return mask


class ObservationEncoder:
    """Encode PyreObservation into a fixed-length float32 vector.

    Mode 'visible': only populate cells within the agent's sight radius —
        mimics true partial observability; preferred for training.
    Mode 'full': expose complete ground-truth grid — useful for debugging
        or oracle upper-bound experiments.

    Output shape: (base_dim,) = padded_grid_features + global_features + wind + difficulty + route hint.

    The 3 extra scalars over the v1 baseline are map-agnostic exit-compass
    features (Fix 3): exit_dx_norm, exit_dy_norm, exit_manhattan_norm.
    These allow the agent to locate the nearest exit on procedurally generated
    maps without having to memorise layout-specific coordinates.

    The 4 newest scalars are a one-hot shortest-path route hint derived from
    the environment metadata: north / south / west / east. This is especially
    useful on procedural hard maps where Manhattan direction can point into a
    wall or dead-end corridor.
    """

    base_dim = (
        MAX_GRID_W * MAX_GRID_H * 10
        + 17
        + len(WINDS)
        + len(DIFFICULTIES)
        + len(DIRECTIONS)
    )

    def __init__(self, mode: str = "visible"):
        if mode not in {"visible", "full"}:
            raise ValueError(f"mode must be 'visible' or 'full', got '{mode}'")
        self.mode = mode

    def encode(self, observation: PyreObservation) -> np.ndarray:
        ms = observation.map_state
        if ms is None:
            raise ValueError("map_state is required for encoding.")

        cell_one_hot = np.zeros((MAX_GRID_H, MAX_GRID_W, 6), dtype=np.float32)
        fire_ch = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        smoke_ch = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        vis_ch = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)
        agent_ch = np.zeros((MAX_GRID_H, MAX_GRID_W), dtype=np.float32)

        visible = {(x, y) for x, y in ms.visible_cells}
        for y in range(ms.grid_h):
            for x in range(ms.grid_w):
                if self.mode == "visible" and (x, y) not in visible and (x, y) != (ms.agent_x, ms.agent_y):
                    continue
                i = y * ms.grid_w + x
                ct = int(ms.cell_grid[i])
                if 0 <= ct <= 5:
                    cell_one_hot[y, x, ct] = 1.0
                fire_ch[y, x] = float(ms.fire_grid[i])
                smoke_ch[y, x] = float(ms.smoke_grid[i])
                vis_ch[y, x] = 1.0 if (x, y) in visible else 0.0

        if 0 <= ms.agent_x < MAX_GRID_W and 0 <= ms.agent_y < MAX_GRID_H:
            agent_ch[ms.agent_y, ms.agent_x] = 1.0

        grid_features = np.concatenate([
            cell_one_hot.reshape(-1),
            fire_ch.reshape(-1),
            smoke_ch.reshape(-1),
            vis_ch.reshape(-1),
            agent_ch.reshape(-1),
        ])

        meta = observation.metadata or {}
        wind = str(meta.get("wind_dir", ms.wind_dir or "CALM")).upper()
        diff = str(meta.get("difficulty", "medium")).lower()
        wi = WINDS.index(wind) if wind in WINDS else 0
        di = DIFFICULTIES.index(diff) if diff in DIFFICULTIES else 1

        wind_oh = np.zeros(len(WINDS), dtype=np.float32); wind_oh[wi] = 1.0
        diff_oh = np.zeros(len(DIFFICULTIES), dtype=np.float32); diff_oh[di] = 1.0
        route_oh = np.zeros(len(DIRECTIONS), dtype=np.float32)
        route_dir = str(meta.get("nearest_exit_direction", "") or "").lower()
        if route_dir in DIRECTIONS:
            route_oh[DIRECTIONS.index(route_dir)] = 1.0

        # Fix 3 — map-agnostic exit compass features.
        # Compute the direction vector and normalised Manhattan distance to the
        # nearest exit cell (cell_type == 4) directly from the live grid.
        # This gives the agent an exit "compass" that works on procedurally
        # generated maps without memorising any layout.
        EXIT_CELL_TYPE = 4
        ax, ay = ms.agent_x, ms.agent_y
        gw, gh = ms.grid_w, ms.grid_h
        best_dist = float(gw + gh)
        best_dx = 0.0
        best_dy = 0.0
        for cy in range(gh):
            for cx in range(gw):
                if int(ms.cell_grid[cy * gw + cx]) == EXIT_CELL_TYPE:
                    d = abs(cx - ax) + abs(cy - ay)
                    if d < best_dist:
                        best_dist = d
                        best_dx = float(cx - ax) / max(1, gw - 1)
                        best_dy = float(cy - ay) / max(1, gh - 1)
        exit_manhattan_norm = best_dist / float(gw + gh)

        global_features = np.array([
            float(observation.agent_health) / 100.0,
            float(ms.agent_health) / 100.0,
            float(ms.step_count) / max(1, ms.max_steps),
            float(ms.fire_spread_rate),
            float(ms.humidity),
            float(ms.agent_x) / max(1, ms.grid_w - 1),
            float(ms.agent_y) / max(1, ms.grid_h - 1),
            float(meta.get("nearest_exit_distance", MAX_GRID_W + MAX_GRID_H) or 0.0) / float(MAX_GRID_W + MAX_GRID_H),
            float(meta.get("reachable_exit_count", 0.0)) / 4.0,
            float(meta.get("visible_cell_count", 0.0)) / float(MAX_GRID_W * MAX_GRID_H),
            float(meta.get("fire_sources", 0.0)) / 5.0,
            {"none": 0.0, "light": 0.33, "moderate": 0.66, "heavy": 1.0}.get(observation.smoke_level, 0.0),
            1.0 if ms.agent_alive else 0.0,
            1.0 if ms.agent_evacuated else 0.0,
            # Fix 3: exit-compass (3 new scalars — map-agnostic, layout-independent)
            best_dx,           # signed x-direction toward nearest exit
            best_dy,           # signed y-direction toward nearest exit
            exit_manhattan_norm,  # how far away the exit is (0 = here, 1 = max)
        ], dtype=np.float32)

        return np.concatenate([grid_features, global_features, wind_oh, diff_oh, route_oh]).astype(np.float32)


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic network for PPO.

    Architecture:
        Input → LayerNorm → FC(512) → LayerNorm → ReLU
                          → FC(256) → LayerNorm → ReLU
                          → FC(128) → ReLU
               ┌──────────────┴──────────────┐
         Policy head (→ logits)        Value head (→ scalar)

    LayerNorm before activations improves gradient flow for the large
    (23,160-dim stacked) flat input without requiring feature normalization.
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        h1, h2, h3 = hidden_sizes

        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )

        # Orthogonal init — standard for PPO (improves early convergence)
        self._init_orthogonal()

        self.policy_head = nn.Linear(h3, action_dim)
        self.value_head = nn.Linear(h3, 1)

        # Small init for output heads prevents saturated softmax early on
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _init_orthogonal(self) -> None:
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """
        Args:
            obs:  (B, input_dim) float32
            mask: (B, action_dim) float32  — 1.0 = valid, 0.0 = invalid
        Returns:
            dist:   Categorical distribution (action masking applied as -inf)
            values: (B,) float32
        """
        features = self.shared(obs)
        logits = self.policy_head(features)

        # Mask invalid actions with -inf before softmax (numerically stable)
        logits = torch.where(mask.bool(), logits, torch.full_like(logits, -1e9))

        dist = torch.distributions.Categorical(logits=logits)
        values = self.value_head(features).squeeze(-1)
        return dist, values

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take greedy) action. Returns (action, log_prob, value)."""
        dist, values = self(obs, mask)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values

    def evaluate(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored actions during PPO update. Returns (log_prob, value, entropy)."""
        dist, values = self(obs, mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, values, entropy


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores transitions for a batch of episodes before PPO update."""
    obs: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear()
        self.masks.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation.

    Returns (returns, advantages) — both shape (T,).
    Episode boundaries (done=True) reset the GAE accumulator so advantages
    don't bleed across episodes within a mixed batch.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            next_value = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return returns, advantages


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    evacuated: bool
    final_health: float
    difficulty: str


def run_episode(
    env: Any,
    network: ActorCritic,
    encoder: ObservationEncoder,
    device: torch.device,
    difficulty: str,
    history_length: int,
    buffer: RolloutBuffer,
    deterministic: bool = False,
    verbose: bool = False,
    step_delay: float = 0.0,
) -> EpisodeResult:
    """Run one episode, appending transitions to *buffer*."""
    observation = env.reset(difficulty=difficulty)
    zero_frame = np.zeros(encoder.base_dim, dtype=np.float32)
    frames: deque = deque([zero_frame.copy() for _ in range(history_length)], maxlen=history_length)
    frames.append(encoder.encode(observation))

    total_reward = 0.0
    final_health = observation.agent_health
    evacuated = False
    steps = 0
    # Anti-loop tracking: remember the last LOOP_WINDOW positions this episode.
    LOOP_WINDOW = 12
    recent_positions: deque = deque(maxlen=LOOP_WINDOW)

    network.eval()
    with torch.no_grad():
        while True:
            state_vec = np.concatenate(list(frames), dtype=np.float32)
            action_mask = build_action_mask(observation)

            obs_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

            action_t, log_prob_t, value_t = network.act(obs_t, mask_t, deterministic=deterministic)

            action_idx = int(action_t.item())
            env_action = action_index_to_env_action(action_idx)

            if step_delay > 0.0:
                time.sleep(step_delay)

            if verbose:
                print(f"    step={steps+1:03d}  action={ACTION_KEYS[action_idx]:<40}  hp={observation.agent_health:5.1f}", flush=True)

            next_obs = env.step(env_action)

            reward = float(next_obs.reward or 0.0)

            # Reward shaping 1 — idle penalty
            chosen_action = env_action.action
            if chosen_action == "wait":
                reward -= 0.05

            # Reward shaping 2 — fire-approach penalty
            ms_next = next_obs.map_state
            if ms_next is not None and chosen_action.startswith("move"):
                ax, ay = ms_next.agent_x, ms_next.agent_y
                gw, gh = ms_next.grid_w, ms_next.grid_h
                fire_grid = ms_next.fire_grid
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < gw and 0 <= ny < gh:
                        if float(fire_grid[ny * gw + nx]) > 0.15:
                            reward -= 0.15
                            break

            # Reward shaping 3 — anti-loop penalty
            if ms_next is not None and chosen_action.startswith("move"):
                cur_pos = (ms_next.agent_x, ms_next.agent_y)
                if cur_pos in recent_positions:
                    reward -= 0.2
                recent_positions.append(cur_pos)

            # Reward shaping 4 — exit proximity pull
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
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    network: ActorCritic,
    optimizer: Adam,
    buffer: RolloutBuffer,
    device: torch.device,
    clip_eps: float,
    value_clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    n_epochs: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """Full PPO update over the collected rollout buffer."""
    rewards = np.array(buffer.rewards, dtype=np.float32)
    values = np.array(buffer.values, dtype=np.float32)
    dones = np.array(buffer.dones, dtype=np.float32)

    returns, advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_arr = torch.tensor(np.stack(buffer.obs), dtype=torch.float32, device=device)
    mask_arr = torch.tensor(np.stack(buffer.masks), dtype=torch.float32, device=device)
    action_arr = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_logp_arr = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    return_arr = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_arr = torch.tensor(advantages, dtype=torch.float32, device=device)
    old_value_arr = torch.tensor(values, dtype=torch.float32, device=device)

    T = len(buffer)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_updates = 0

    network.train()
    for _ in range(n_epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, minibatch_size):
            idx = perm[start:start + minibatch_size]
            if len(idx) < 2: continue

            log_prob, value, entropy = network.evaluate(obs_arr[idx], mask_arr[idx], action_arr[idx])
            ratio = torch.exp(log_prob - old_logp_arr[idx])
            adv_mb = adv_arr[idx]
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            ret_mb = return_arr[idx]
            old_val_mb = old_value_arr[idx]
            value_pred_clipped = old_val_mb + torch.clamp(value - old_val_mb, -value_clip_eps, value_clip_eps)
            value_loss = torch.max(F.mse_loss(value, ret_mb), F.mse_loss(value_pred_clipped, ret_mb))

            entropy_loss = -entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - (log_prob - old_logp_arr[idx])).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.mean().item()
            metrics["approx_kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            n_updates += 1

    if n_updates > 0:
        for k in metrics: metrics[k] /= n_updates
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    env: Any,
    network: ActorCritic,
    encoder: ObservationEncoder,
    device: torch.device,
    difficulty: str,
    history_length: int,
    n_episodes: int,
) -> Dict[str, float]:
    rewards, successes, steps = [], [], []
    dummy_buffer = RolloutBuffer()
    for _ in range(n_episodes):
        result = run_episode(
            env=env, network=network, encoder=encoder, device=device,
            difficulty=difficulty, history_length=history_length,
            buffer=dummy_buffer, deterministic=True,
        )
        dummy_buffer.clear()
        rewards.append(result.total_reward)
        successes.append(float(result.evacuated))
        steps.append(result.steps)
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_max": float(np.max(rewards)),
        "success_rate": float(np.mean(successes)),
        "steps_mean": float(np.mean(steps)),
    }


# ---------------------------------------------------------------------------
# PNG graph (matplotlib)
# ---------------------------------------------------------------------------

def save_training_graph_png(
    path: Path,
    episode_rows: List[Dict],
    eval_rows: List[Dict],
    window: int = 20,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[warn] matplotlib not installed — skipping PNG graph.")
        return

    if not episode_rows: return
    path.parent.mkdir(parents=True, exist_ok=True)

    episodes   = [int(r["episode"]) for r in episode_rows]
    rewards    = [float(r["reward"]) for r in episode_rows]
    evacuated  = [float(r["evacuated"]) for r in episode_rows]
    difficulty = [str(r["difficulty"]) for r in episode_rows]

    def ma(values: list, w: int) -> list:
        out, run, q = [], 0.0, []
        for v in values:
            q.append(v); run += v
            if len(q) > w: run -= q.pop(0)
            out.append(run / len(q))
        return out

    reward_ma  = ma(rewards, window)
    success_ma = ma(evacuated, window)
    eval_eps  = [int(r["episode"]) for r in eval_rows]
    eval_succ = [float(r["success_rate"]) for r in eval_rows]

    diff_colors = {"easy": "#d4edda", "medium": "#fff3cd", "hard": "#f8d7da"}
    regions = []
    if difficulty:
        cur, start = difficulty[0], episodes[0]
        for ep, d in zip(episodes[1:], difficulty[1:]):
            if d != cur:
                regions.append((start, ep, cur))
                cur, start = d, ep
        regions.append((start, episodes[-1], cur))

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for x0, x1, diff in regions:
        ax1.axvspan(x0, x1, color=diff_colors.get(diff, "#eeeeee"), alpha=0.35, zorder=0)

    ax1.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax1.plot(episodes, rewards, color="#d1c7bc", linewidth=0.8, alpha=0.6, label="Episode reward", zorder=2)
    ax1.plot(episodes, reward_ma, color="#c1661c", linewidth=2.5, label=f"Reward (MA-{window})", zorder=3)
    ax2.plot(episodes, success_ma, color="#1a7a8a", linewidth=2.5, label=f"Success rate (MA-{window})", zorder=3)

    if eval_eps:
        ax2.scatter(eval_eps, eval_succ, color="#0d5b6b", s=60, zorder=5, marker="D", label="Eval success", edgecolors="white", linewidths=1.2)

    ax1.set_xlabel("Episode", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Reward", fontsize=13, fontweight="bold", color="#c1661c")
    ax2.set_ylabel("Success Rate", fontsize=13, fontweight="bold", color="#1a7a8a")
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax1.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.8)
    ax1.set_xlim(episodes[0], episodes[-1])

    total_eps = episodes[-1]
    final_sr  = success_ma[-1] if success_ma else 0.0
    fig.suptitle(f"Pyre PPO Training  —  {total_eps} episodes  |  final success rate: {final_sr:.0%}", fontsize=14, fontweight="bold")

    import matplotlib.patches as mpatches
    diff_patches = [mpatches.Patch(color=diff_colors[d], alpha=0.6, label=d.capitalize()) for d in ["easy", "medium", "hard"] if any(r == d for r in difficulty)]
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2 + diff_patches, l1 + l2 + [p.get_label() for p in diff_patches], loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Curriculum scheduling
# ---------------------------------------------------------------------------

def build_curriculum(schedule_str: str, n_episodes: int) -> List[str]:
    stages = [s.strip().lower() for s in schedule_str.split(",") if s.strip()]
    if not stages: stages = ["medium"]
    for s in stages:
        if s not in DIFFICULTIES: raise ValueError(f"Unknown difficulty '{s}'.")
    seg = max(1, n_episodes // len(stages))
    schedule = []
    for s in stages: schedule.extend([s] * seg)
    while len(schedule) < n_episodes: schedule.append(stages[-1])
    return schedule[:n_episodes]


def parse_mix_dist(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec: return None
    out = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        if ":" not in chunk: raise ValueError(f"Invalid mix-dist entry '{chunk}'")
        name, val = chunk.split(":", 1)
        out[name.strip().lower()] = float(val)
    total = sum(out.values())
    if total <= 0: raise ValueError(f"mix-dist probabilities must be positive")
    return {k: v / total for k, v in out.items()}


class PatienceCurriculum:
    def __init__(self, stages: List[str], threshold: float, patience_window: int, mix_ratio: float = 0.0, mix_dist: Optional[Dict[str, float]] = None):
        self.stages, self.threshold, self.patience_window = stages, threshold, patience_window
        self.mix_ratio, self.mix_dist = mix_ratio, mix_dist
        self.stage_idx = 0
        self._streak = 0
        if self.mix_dist is not None:
            for k in self.mix_dist:
                if k not in self.stages: raise ValueError(f"mix_dist key '{k}' not in stages")

    @property
    def current(self) -> str: return self.stages[self.stage_idx]

    def step(self, success_rate_30: float) -> str:
        if self.stage_idx < len(self.stages) - 1:
            if success_rate_30 >= self.threshold: self._streak += 1
            else: self._streak = 0
            if self._streak >= self.patience_window:
                self.stage_idx += 1
                self._streak = 0
                print(f"  [curriculum] Advanced to '{self.current}' (suc30={success_rate_30:.2f})")
        is_final = self.stage_idx == len(self.stages) - 1
        if is_final and self.mix_dist is not None:
            keys = list(self.mix_dist.keys())
            probs = np.array([self.mix_dist[k] for k in keys])
            return str(np.random.choice(keys, p=probs / probs.sum()))
        if is_final and self.mix_ratio > 0.0 and len(self.stages) >= 2:
            if np.random.rand() < self.mix_ratio: return self.stages[self.stage_idx - 1]
        return self.current


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, network: ActorCritic, optimizer: Adam, scheduler, episode: int, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"episode": episode, "network_state": network.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict() if scheduler else None, "args": vars(args)}, path)

def load_checkpoint(path: Path, network: ActorCritic, optimizer: Adam, scheduler) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    network.load_state_dict(ckpt["network_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"): scheduler.load_state_dict(ckpt["scheduler_state"])
    start_ep = int(ckpt.get("episode", 0))
    print(f"[resume] Loaded checkpoint from episode {start_ep}")
    return start_ep


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.server:
        env = HttpPyreEnv(base_url=args.server)
        if not env.health_check(): sys.exit(f"[error] Server not reachable at {args.server}")
    else:
        if PyreEnvironment is None: sys.exit("[error] PyreEnvironment not found. Use --server.")
        env = PyreEnvironment(max_steps=args.max_steps)

    encoder = ObservationEncoder(mode=args.observation_mode)
    input_dim = encoder.base_dim * args.history_length
    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))
    network = ActorCritic(input_dim=input_dim, action_dim=ACTION_DIM, hidden_sizes=hidden_sizes).to(device)
    optimizer = Adam(network.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=max(1, args.episodes // args.update_every)) if args.lr_decay else None

    stages = [s.strip().lower() for s in args.difficulty_schedule.split(",") if s.strip()]
    if args.patience_threshold > 0:
        curriculum = PatienceCurriculum(stages=stages, threshold=args.patience_threshold, patience_window=args.patience_window, mix_ratio=args.hard_mix_ratio, mix_dist=parse_mix_dist(getattr(args, "hard_mix_dist", None)))
        static_curriculum = None
    else:
        curriculum = None
        static_curriculum = build_curriculum(args.difficulty_schedule, args.episodes)

    start_ep = 0
    if args.resume and Path(args.resume).exists():
        start_ep = load_checkpoint(Path(args.resume), network, optimizer, scheduler)

    buffer = RolloutBuffer()
    episode_rows, eval_rows = [], []
    reward_window, success_window = deque(maxlen=30), deque(maxlen=30)
    t_start = time.time()
    next_ep_diff = curriculum.stages[0] if curriculum else None

    for ep_idx in range(start_ep, args.episodes):
        play_diff = next_ep_diff if curriculum else static_curriculum[ep_idx]
        ep_num = ep_idx + 1
        ep_step_delay = args.step_delay if ep_num > args.viz_after_ep else 0.0

        result = run_episode(env=env, network=network, encoder=encoder, device=device, difficulty=play_diff, history_length=args.history_length, buffer=buffer, verbose=args.verbose, step_delay=ep_step_delay)
        reward_window.append(result.total_reward); success_window.append(float(result.evacuated))
        if curriculum: next_ep_diff = curriculum.step(float(np.mean(success_window)))

        episode_rows.append({"episode": ep_num, "difficulty": play_diff, "reward": round(result.total_reward, 4), "evacuated": int(result.evacuated), "steps": result.steps, "hp": round(result.final_health, 2), "r30": round(float(np.mean(reward_window)), 4), "s30": round(float(np.mean(success_window)), 4)})
        print(f"ep={ep_num:04d} [{play_diff:<6}] steps={result.steps:03d} reward={result.total_reward:+8.3f} evac={int(result.evacuated)} hp={result.final_health:5.1f} s30={float(np.mean(success_window)):.2f} t={time.time()-t_start:.0f}s")

        if (ep_num % args.update_every == 0) or (ep_num == args.episodes):
            metrics = ppo_update(network, optimizer, buffer, device, args.clip_eps, args.clip_eps, args.entropy_coef, args.value_coef, args.update_epochs, args.minibatch_size, args.gamma, args.gae_lambda, args.max_grad_norm)
            if scheduler: scheduler.step()
            buffer.clear()
            print(f"  >> PPO update pi_loss={metrics['policy_loss']:+.4f} v_loss={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f} kl={metrics['approx_kl']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if args.eval_every > 0 and (ep_num % args.eval_every == 0 or ep_num == args.episodes):
            eval_m = evaluate_policy(env, network, encoder, device, args.eval_difficulty, args.history_length, args.eval_episodes)
            eval_rows.append({"episode": ep_num, "difficulty": args.eval_difficulty, **{k: round(v, 4) for k, v in eval_m.items()}})
            print(f"  ** EVAL [{args.eval_difficulty}] reward={eval_m['reward_mean']:+.3f} success={eval_m['success_rate']:.2f}")

        if args.checkpoint and args.checkpoint_every > 0 and ep_num % args.checkpoint_every == 0:
            save_checkpoint(Path(args.checkpoint), network, optimizer, scheduler, ep_num, args)

    if args.output:
        out = Path(args.output)
        save_checkpoint(out, network, optimizer, scheduler, args.episodes, args)
        if args.save_metrics:
            save_csv(out.with_suffix(".csv"), episode_rows)
            if eval_rows: save_csv(out.parent / (out.stem + "_eval.csv"), eval_rows)
        if args.save_graph: save_training_graph_png(out.with_suffix(".png"), episode_rows, eval_rows)

def save_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--step-delay", type=float, default=0.0)
    p.add_argument("--viz-after-ep", type=int, default=1000000)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--difficulty-schedule", type=str, default="easy,medium,hard_fixed,hard")
    p.add_argument("--patience-threshold", type=float, default=0.65)
    p.add_argument("--patience-window", type=int, default=15)
    p.add_argument("--hard-mix-ratio", type=float, default=0.25)
    p.add_argument("--hard-mix-dist", type=str, default=None)
    p.add_argument("--eval-difficulty", type=str, default="medium")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--observation-mode", type=str, default="visible", choices=("visible", "full"))
    p.add_argument("--history-length", type=int, default=4)
    p.add_argument("--hidden-sizes", type=str, default="512,256,128")
    p.add_argument("--update-every", type=int, default=5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-decay", action="store_true", default=True)
    p.add_argument("--lr-end-factor", type=float, default=0.1)
    p.add_argument("--output", type=str, default="artifacts/pyre_ppo.pt")
    p.add_argument("--checkpoint", type=str, default="artifacts/pyre_ppo_checkpoint.pt")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-metrics", action="store_true", default=True)
    p.add_argument("--save-graph", action="store_true", default=True)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--describe-only", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.describe_only: print(__doc__); sys.exit(0)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    log_tee = TeeLogger(sys.stdout, Path(args.log_file)) if args.log_file else None
    if log_tee: sys.stdout = log_tee
    try: train(args)
    finally:
        if log_tee: log_tee.close()
