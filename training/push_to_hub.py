"""
push_to_hub.py — Upload Pyre PPO training artifacts to the Hugging Face Hub.

Uploads files resolved by ``--stem`` (e.g. ``pyre_ppo_fixed.pti`` → Hub ``model.pt``).

Also uploads a **hardcoded** model card (README.md) documenting the
``pyre_ppo_fixed`` HTTP training run (metrics taken from ``artifacts/``).

Usage
─────
  python training/push_to_hub.py \\
      --repo-id your-hf-username/pyre-ppo-agent \\
      --stem pyre_ppo_fixed \\
      --token $HF_TOKEN

  # Private repo:
  python training/push_to_hub.py --repo-id krooz/pyre-ppo-agent --stem pyre_ppo_fixed --private --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

# Colab link (same as pyre_env/README.md — Training section)
PYRE_PPO_COLAB_URL = (
    "https://colab.research.google.com/drive/1ojC55qKXMVRXdjKeG5dUHiA5RBOBxA9V?usp=sharing"
)


def build_hub_model_card(repo_id: str, model_filename: str = "model.pt") -> str:
    """Hardcoded README for the ``pyre_ppo_fixed`` run (from ``artifacts/pyre_ppo_fixed_*``)."""
    return f"""---
tags:
  - openenv
  - reinforcement-learning
  - ppo
  - pyre
  - fire-evacuation
license: mit
---

# Pyre PPO Agent — `{repo_id}`

PPO-trained actor-critic agent for the [Pyre](https://huggingface.co/spaces/Krooz/pyre_env)
fire-evacuation environment (OpenEnv Hackathon, Apr 2026).

> ⚠️ This is a raw PyTorch checkpoint, **not** a `transformers` model.
> The Hugging Face hosted Inference API cannot run it directly.
> Use the inference code below to load and run it locally.

## Training summary (artifact run: ``pyre_ppo_fixed``)

Values below are from ``artifacts/pyre_ppo_fixed.csv``, ``pyre_ppo_fixed_eval.csv``,
and ``artifacts/pyre_ppo_fixed_training.log`` (HTTP trainer, env server at ``http://localhost:8000``).

| Metric | Value |
|--------|-------|
| Total episodes | **200** |
| Wall-clock training time | **~48 s** (~4.2 eps/s on CPU) |
| Final success rate (rolling last 30 ep) | **80%** |
| Final reward mean (rolling last 30 ep) | **+8.446** |
| Curriculum | **Static** ``easy,medium`` (≈100 eps each; ``--patience-threshold 0``) |
| Eval cadence | Every **20** episodes, **3** deterministic rollouts |
| Eval difficulty | **medium** (per eval log / ``pyre_ppo_fixed_eval.csv``) |

## Network architecture (from training log)

| Property | Value |
|----------|-------|
| Total parameters | **12,065,650** |
| Input vector dim | **23,140** (encoder ``base_dim`` 5785 × **4** stacked frames) |
| Action dim | **41** (4 move + 4 look + 1 wait + 16 door open + 16 door close) |
| Hidden MLP | **512 → 256 → 128** |

## Hyperparameters (defaults matching this run)

| Param | Value |
|-------|-------|
| Learning rate | **3×10⁻⁴** |
| PPO clip ε | **0.2** |
| Entropy coeff | **0.03** |
| Value coeff | **0.5** |
| Gamma | **0.99** |
| GAE λ | **0.95** |
| PPO update every | **5** episodes |
| PPO epochs / minibatch | **4** / **256** |
| Max grad norm | **0.5** |
| Observation mode | **visible** (partial observability) |
| Device | **cpu** |

### Evaluation checkpoints (from ``pyre_ppo_fixed_eval.csv``)

| Episode | Difficulty | Success rate | Reward mean | Steps mean |
|---------|------------|--------------|-------------|------------|
| 20 | medium | 100% | +15.698 | 7.0 |
| 40 | medium | 100% | +15.640 | 4.3 |
| 60 | medium | 100% | +16.887 | 9.0 |
| 80 | medium | 100% | +15.162 | 10.3 |
| 100 | medium | 67% | +6.008 | 57.0 |
| 120 | medium | 67% | +6.401 | 32.7 |
| 140 | medium | 100% | +16.283 | 6.3 |
| 160 | medium | 100% | +16.573 | 8.3 |
| 180 | medium | 100% | +16.397 | 8.0 |
| 200 | medium | 67% | +6.807 | 14.7 |

## Files in this repository

| File | Description |
|------|-------------|
| `{model_filename}` | PyTorch checkpoint (`network_state`, `optimizer_state`, `scheduler_state`, `args`, `episode`) |
| `training_graph.png` | Training curves (reward + success rate vs episode) |
| `episode_metrics.csv` | Per-episode training metrics |
| `eval_metrics.csv` | Periodic eval aggregates |
| `training.log` | Full console transcript of the HTTP training run |

## Running inference locally

```python
import sys
import torch
from huggingface_hub import hf_hub_download

# 1. Point Python at your local pyre_env checkout (or install the package)
sys.path.insert(0, "pyre_env")

from training.ppo.train_torch_ppo import (
    ActorCritic,
    ObservationEncoder,
    action_index_to_env_action,
    build_action_mask,
)

# 2. Download the checkpoint from this Hub repo
ckpt_path = hf_hub_download(repo_id="{repo_id}", filename="{model_filename}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 3. Rebuild the policy from saved training args
saved_args = ckpt["args"]
encoder = ObservationEncoder(mode=saved_args.get("observation_mode", "visible"))
hidden_sizes = tuple(int(x) for x in saved_args.get("hidden_sizes", "512,256,128").split(","))
history_length = saved_args.get("history_length", 4)
input_dim = encoder.base_dim * history_length
network = ActorCritic(input_dim, 41, hidden_sizes)
network.load_state_dict(ckpt["network_state"])
network.eval()
print(f"Loaded checkpoint from episode {{ckpt.get('episode', '?')}}")

# 4. Roll out one episode (in-process env — swap for HTTP client if you prefer)
from openenv_pyre import PyreEnvironment
from collections import deque
import numpy as np

env = PyreEnvironment()
obs = env.reset(difficulty="medium")
frames = deque([np.zeros(encoder.base_dim, dtype=np.float32)] * history_length, maxlen=history_length)
frames.append(encoder.encode(obs))

total_reward = 0.0
with torch.no_grad():
    while True:
        state_vec = np.concatenate(list(frames), dtype=np.float32)
        obs_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(build_action_mask(obs, exclude_look=True), dtype=torch.float32).unsqueeze(0)
        action_t, _, _ = network.act(obs_t, mask_t, deterministic=True)
        obs = env.step(action_index_to_env_action(int(action_t.item())))
        total_reward += float(obs.reward or 0.0)
        frames.append(encoder.encode(obs))
        if obs.done:
            break

print(f"Episode finished — evacuated={{obs.agent_evacuated}}  reward={{total_reward:.3f}}")
```

## Environment & training resources

- **HF Space (live env)**: [Krooz/pyre_env](https://huggingface.co/spaces/Krooz/pyre_env)
- **PPO training in Colab (HTTP to Space)**: [Pyre PPO training — Google Colab]({PYRE_PPO_COLAB_URL})
- **Local HTTP trainer**: ``training/ppo/train_torch_ppo_http.py``
- **Local in-process trainer**: ``training/ppo/train_torch_ppo.py``
- **Notebook source**: ``training/ppo/pyre_ppo_training.ipynb``
"""


def _resolve_model_file(artifacts_dir: Path, stem: str) -> Optional[Path]:
    """Find the model file for a given stem — tries .pt then .pti extensions."""
    for ext in (".pt", ".pti"):
        p = artifacts_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def push(
    repo_id: str,
    artifacts_dir: Path,
    token: str,
    stem: str = "pyre_ppo",
    private: bool = False,
    commit_message: str = "Upload Pyre PPO training artifacts",
) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Install with:  pip install huggingface_hub"
        )

    api = HfApi(token=token)

    print(f"[hub] Creating / validating repo: {repo_id}  (private={private})")
    create_repo(repo_id=repo_id, repo_type="model", private=private,
                exist_ok=True, token=token)

    # Resolve model file — accepts both .pt and .pti
    model_path = _resolve_model_file(artifacts_dir, stem)
    model_hub_name = "model.pt"

    # Map local paths → canonical hub filenames
    candidates: list[tuple[Optional[Path], str, str]] = [
        (model_path,                                   model_hub_name,        "model checkpoint"),
        (artifacts_dir / f"{stem}.png",                "training_graph.png",  "training graph"),
        (artifacts_dir / f"{stem}.csv",                "episode_metrics.csv", "episode metrics"),
        (artifacts_dir / f"{stem}_eval.csv",           "eval_metrics.csv",    "eval metrics"),
        (artifacts_dir / f"{stem}_training.log",       "training.log",        "training log"),
    ]

    uploaded: list[str] = []
    for path, hub_name, label in candidates:
        if path is not None and path.exists():
            size_kb = path.stat().st_size // 1024
            size_str = f"{size_kb // 1024} MB" if size_kb > 1024 else f"{size_kb} KB"
            print(f"[hub] Uploading {path.name} → {hub_name}  ({label})  [{size_str}]")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=hub_name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {label}",
                token=token,
            )
            uploaded.append(hub_name)
        else:
            local = path.name if path else f"{stem}.(pt|pti)"
            print(f"[hub] Skipping {local} — not found in {artifacts_dir}")

    # Build and upload model card (hardcoded metrics for the documented artifact run)
    card = build_hub_model_card(repo_id, model_filename=model_hub_name)

    card_path = artifacts_dir / "README_hub.md"
    card_path.write_text(card, encoding="utf-8")
    print("[hub] Uploading model card (README.md)")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
        token=token,
    )
    card_path.unlink()  # clean up temp file

    print(f"\n✓ Done. View at: https://huggingface.co/{repo_id}")
    print(f"  Uploaded: {', '.join(uploaded) or 'none'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push Pyre PPO training artifacts to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-id", required=True,
                   help="HF Hub repo id, e.g. 'krooz/pyre-ppo-agent'")
    p.add_argument("--artifacts-dir", type=str, default="artifacts",
                   help="Directory containing the training artifacts")
    p.add_argument("--stem", type=str, default="pyre_ppo",
                   help="Filename stem used when training, e.g. 'pyre_ppo_fixed' if your "
                        "files are pyre_ppo_fixed.pti / pyre_ppo_fixed.png / etc.")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--private", action="store_true",
                   help="Create a private repository")
    p.add_argument("--commit-message", type=str, default="Upload Pyre PPO training artifacts")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "No HuggingFace token found.\n"
            "Pass --token YOUR_TOKEN  or  export HF_TOKEN=YOUR_TOKEN"
        )

    push(
        repo_id=args.repo_id,
        artifacts_dir=Path(args.artifacts_dir),
        stem=args.stem,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )
