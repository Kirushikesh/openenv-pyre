"""
train_grpo_openenv.py — GRPO + LoRA training for Pyre
                        (PyreEnvironment in-process, OpenEnv notebook-style pipeline)

Mirrors train_grpo_openenv.py from the reference as closely as possible.
Differences vs. the reference are intentional and limited to:

  1. Environment is PyreEnvironment, called in-process (no server needed).
     The reference calls WorldState in-process; Pyre uses the same approach —
     PyreEnvironment is instantiated directly, so no `uv run server` is required.

  2. Pyre is a single-environment, difficulty-parametrised task.  For this initial
     training run we fix difficulty to "easy" only (1 fire source, slow spread,
     calm wind, high humidity) — the clearest training signal before adding harder
     difficulties.  The stratified-mix infrastructure is kept so additional levels
     can be added by expanding DIFFICULTY_REGISTRY and DIFFICULTY_WEIGHTS.

  3. Prompt & completion lengths are similar to the reference:
       - Pyre observations are first-person narrative (~200-300 tokens per step).
       - Outputs are <think>…</think> + one small JSON action → up to ~512 tokens.
     We use max_completion_length=512 (action JSON is tiny vs. IAM tool calls).
     Everything else (lr, grad_accum, num_generations, vllm colocate, trackio,
     push_to_hub) matches the reference notebook.

  4. The reward signal is NOT passed to the model during training.  Evals-mode
     (`evals.py`) appends `reward: {r:+.3f}` to each history step and includes a
     "REWARD SIGNAL" section in the system prompt so an untrained model can use
     the signal as a hint.  During GRPO training the model must learn from the
     reward functions alone — leaking the reward into the prompt would contaminate
     the gradient signal.

Usage
─────
  python train_grpo_openenv.py
  python train_grpo_openenv.py --model-id Qwen/Qwen3-1.7B --dataset-size 1000

Dependencies
────────────
  uv sync   (installs trl, transformers, datasets, openenv-core, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# PyreEnv client — imported from the same package
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pyre_env import PyreAction  # noqa: E402
from pyre_env.server.pyre_env_environment import PyreEnvironment  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo_openenv")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Difficulty registry + stratified mix (notebook "cell-1" analogue)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DifficultyMeta:
    difficulty: str
    max_steps: int
    optimal_steps: int   # H* — used by reward_efficiency


DIFFICULTY_REGISTRY: Dict[str, DifficultyMeta] = {
    # Only "easy" for this initial training run.
    # Add "medium" / "hard" here and update DIFFICULTY_WEIGHTS to scale up.
    "easy": DifficultyMeta("easy", max_steps=200, optimal_steps=50),
}

# Stratified mix — weights must sum to 1.0.
DIFFICULTY_WEIGHTS: Dict[str, float] = {
    "easy": 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. System prompt  (reward signal intentionally omitted — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

system_prompt = textwrap.dedent("""
You are an agent trapped inside a burning building.
Your goal is to navigate to an EXIT before your health reaches zero or time runs out.

ENVIRONMENT RULES
- You have partial vision: you cannot see through walls or dense smoke.
- Fire and smoke spread each step — do NOT linger in hazardous areas.
- Closing a door adjacent to active fire slows its spread (strategic move).
- Your health drains faster in moderate/heavy smoke and on fire cells.
- Exits may be BLOCKED if fire burns directly on them — check available hints.

OUTPUT FORMAT (STRICT)
You MUST reason inside <think>...</think> tags first, then emit EXACTLY ONE JSON object.
Output NOTHING else — no extra text, no markdown fences, no second JSON block.

<think>
Brief reasoning: what can I see, where is danger, what is the safest next move?
</think>
{"action": "move", "direction": "north"}

AVAILABLE ACTIONS
- move  : {"action": "move", "direction": "north|south|east|west"}
- look  : {"action": "look", "direction": "north|south|east|west"}   ← scan 5 cells ahead
- door  : {"action": "door", "target_id": "door_X", "door_state": "open|close"}
- wait  : {"action": "wait"}

STRATEGY TIPS
- Use `look` to scout a direction before entering an unknown corridor.
- Closing a door between you and fire buys time; re-open when clear.
- If smoke is heavy, back away; your health drains fast in thick smoke.
- Door IDs (e.g. door_3) appear in the Visible objects list — use them with the door action.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper functions (notebook "cell-10" analogue)
# ─────────────────────────────────────────────────────────────────────────────

def make_user_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    """Build the user turn from the current PyreObservation dict + history.

    Reward is intentionally excluded from both the header and history entries.
    The model must learn from GRPO reward functions, not from in-prompt leakage.
    """
    narrative = obs.get("narrative", "(no narrative)")
    # Strip the "Available actions:" line the narrative builder appends —
    # the system prompt already documents the full action schema.
    narrative = re.sub(r"\nAvailable actions:.*$", "", narrative, flags=re.MULTILINE)

    health       = obs.get("agent_health", 0.0)
    health_st    = obs.get("health_status", "?")
    location     = obs.get("location_label", "?")
    smoke        = obs.get("smoke_level", "none")
    fire_vis     = obs.get("fire_visible", False)
    fire_dir     = obs.get("fire_direction") or "none"
    wind         = obs.get("wind_dir", "CALM")
    elapsed      = obs.get("elapsed_steps", 0)
    blocked_exits = obs.get("blocked_exit_ids", [])
    visible_objs  = obs.get("visible_objects", [])
    audible       = obs.get("audible_signals", [])

    status_line = (
        f"Health: {health:.1f} | Status: {health_st} | Location: {location}\n"
        f"Smoke: {smoke} | Fire visible: {fire_vis}"
        + (f" (direction: {fire_dir})" if fire_vis else "")
        + f"\nWind: {wind} | Steps elapsed: {elapsed}"
    )
    if blocked_exits:
        status_line += f"\nBLOCKED exits (fire on them): {', '.join(blocked_exits)}"
    if visible_objs:
        obj_strs = [
            f"{o.get('type','?')} '{o.get('id','?')}' {o.get('relative_pos','?')}"
            + (f" [{o.get('state','')}]" if o.get("state") else "")
            for o in visible_objs
        ]
        status_line += f"\nVisible objects: {'; '.join(obj_strs)}"
    if audible:
        status_line += f"\nSounds: {'; '.join(audible)}"

    history_str = ""
    if history:
        recent = history[-8:]
        history_str = (
            "=== RECENT ACTION HISTORY (action → feedback → health) ===\n"
            + "\n".join(recent) + "\n\n"
        )

    return (
        f"=== CURRENT OBSERVATION ===\n{narrative}\n\n"
        f"=== STATUS ===\n{status_line}\n\n"
        + history_str
        + "What is your next action? Respond with <think>...</think> then a single JSON action."
    )


_VALID_ACTIONS     = {"move", "door", "look", "wait"}
_VALID_DIRECTIONS  = {"north", "south", "east", "west"}
_VALID_DOOR_STATES = {"open", "close"}
_FALLBACK_ACTION   = {"action": "wait"}


def _validate_pyre_action(blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action = blob.get("action", "").strip().lower()
    if action not in _VALID_ACTIONS:
        return None
    out: Dict[str, Any] = {"action": action}
    if action in ("move", "look"):
        direction = str(blob.get("direction", "")).strip().lower()
        if direction not in _VALID_DIRECTIONS:
            return None
        out["direction"] = direction
    elif action == "door":
        tid = blob.get("target_id", "") or blob.get("target", "")
        ds  = str(blob.get("door_state", "")).strip().lower()
        if not tid or ds not in _VALID_DOOR_STATES:
            return None
        out["target_id"]  = str(tid)
        out["door_state"] = ds
    return out


def parse_action(text: str) -> Tuple[Dict[str, Any], float]:
    """Extract a Pyre action from raw LLM text.

    Returns (action_dict, format_score):
      1.0  — valid JSON + <think> tags
      0.7  — valid JSON, no <think>
      0.4  — partial JSON rescued via regex
      0.1  — action keyword found in raw text (last resort)
      0.0  — completely unparseable → {"action": "wait"} fallback
    """
    has_think = "<think>" in text and "</think>" in text

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            blob = json.loads(text[start:end + 1])
            if isinstance(blob, dict):
                action = _validate_pyre_action(blob)
                if action is not None:
                    return action, (1.0 if has_think else 0.7)
        except json.JSONDecodeError:
            pass

    for m in re.finditer(r'\{[^{}]+\}', text):
        try:
            blob = json.loads(m.group())
            if isinstance(blob, dict) and "action" in blob:
                action = _validate_pyre_action(blob)
                if action is not None:
                    return action, 0.4
        except json.JSONDecodeError:
            continue

    lower = text.lower()
    for d in _VALID_DIRECTIONS:
        if f"move {d}" in lower:
            return {"action": "move", "direction": d}, 0.1
    for d in _VALID_DIRECTIONS:
        if f"look {d}" in lower:
            return {"action": "look", "direction": d}, 0.1
    door_m = re.search(r'door[_\s]*([\w]+)', lower)
    if door_m:
        tid = door_m.group(1)
        ds  = "close" if "clos" in lower else "open"
        return {"action": "door", "target_id": tid, "door_state": ds}, 0.1
    if "wait" in lower:
        return {"action": "wait"}, 0.1

    return dict(_FALLBACK_ACTION), 0.0


print("Helper functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rollout function (notebook "cell-12" analogue)
# ─────────────────────────────────────────────────────────────────────────────

def rollout_once(
    trainer: GRPOTrainer,
    tokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
    difficulty: str,
    seed: int,
) -> Dict[str, Any]:
    """Execute one full Pyre episode using generate_rollout_completions.

    Uses PyreEnv.sync() (WebSocket context) so the session persists across all
    steps of the episode — identical semantics to WorldState in the reference.
    """
    prompt_ids:     List[int]   = []
    completion_ids: List[int]   = []
    logprobs:       List        = []
    step_rewards:   List[float] = []
    history:        List[str]   = []
    agent_evacuated: bool       = False
    final_health:    float      = 0.0
    done:            bool       = False
    steps_taken:     int        = 0
    think_steps:     int        = 0

    MAX_TOK_ACCUM = 8192

    try:
        env  = PyreEnvironment()
        obs  = env.reset(difficulty=difficulty, seed=seed)
        done = obs.done

        for _turn in range(max_turns):
            if done or len(completion_ids) >= MAX_TOK_ACCUM:
                break

            obs_dict = obs.model_dump()
            user_prompt = make_user_prompt(obs_dict, history)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]

            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=True,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )

            log.debug(
                "\n[difficulty=%s seed=%d step=%d] ── PROMPT ──────────────────────────\n%s\n────────────────────────────────────",
                difficulty, seed, _turn + 1, prompt_text,
            )

            rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
            prompt_ids.extend(rollout_outputs["prompt_ids"])
            completion_ids.extend(rollout_outputs["completion_ids"])
            logprobs.extend(rollout_outputs.get("logprobs") or [])
            completion_text = rollout_outputs.get("text") or tokenizer.decode(
                rollout_outputs["completion_ids"], skip_special_tokens=False
            )

            log.debug(
                "\n[difficulty=%s seed=%d step=%d] ── COMPLETION ──────────────────────\n%s\n────────────────────────────────────",
                difficulty, seed, _turn + 1, completion_text,
            )

            if "<think>" in completion_text and "</think>" in completion_text:
                think_steps += 1
            steps_taken += 1

            action_dict, fmt_score = parse_action(completion_text)
            fmt_penalty = (1.0 - fmt_score) * -0.10

            obs         = env.step(PyreAction(**action_dict))
            step_reward = float(obs.reward or 0.0)
            done        = obs.done

            step_rewards.append(step_reward + fmt_penalty)

            feedback = obs.last_action_feedback or ""
            # Reward is intentionally excluded from the history entry.
            history.append(
                f"Step {_turn + 1}: {json.dumps(action_dict)}"
                + (f"\n  → {feedback}" if feedback else "")
                + f"\n  health: {obs.agent_health:.1f}"
            )

        agent_evacuated = obs.agent_evacuated
        final_health    = float(obs.agent_health)

    except Exception as exc:
        log.error("Episode failed (difficulty=%s seed=%d): %s", difficulty, seed, exc)
        return {
            "prompt_ids":       [],
            "completion_ids":   [],
            "logprobs":         [],
            "evacuated":        0,
            "final_health":     0.0,
            "mean_step_reward": 0.0,
            "steps_taken":      0,
            "optimal_steps":    DIFFICULTY_REGISTRY[difficulty].optimal_steps,
            "format_rate":      0.0,
        }

    mean_step_reward = sum(step_rewards) / max(len(step_rewards), 1)
    format_rate      = think_steps / max(steps_taken, 1)
    meta             = DIFFICULTY_REGISTRY[difficulty]

    return {
        "prompt_ids":       prompt_ids,
        "completion_ids":   completion_ids,
        "logprobs":         logprobs,
        "evacuated":        int(agent_evacuated),
        "final_health":     round(final_health, 2),
        "mean_step_reward": round(mean_step_reward, 4),
        "steps_taken":      steps_taken,
        "optimal_steps":    meta.optimal_steps,
        "format_rate":      round(format_rate, 4),
    }


def rollout_func(prompts, trainer=None):
    """Called by GRPOTrainer once per training batch."""
    episode_prompt_ids:     List[List[int]]   = []
    episode_completion_ids: List[List[int]]   = []
    episode_logprobs:       List[List[float]] = []
    evacuated_list:         List[int]         = []
    final_health_list:      List[float]       = []
    step_reward_means:      List[float]       = []
    steps_taken_list:       List[int]         = []
    optimal_steps_list:     List[int]         = []
    format_rates:           List[float]       = []

    difficulty_ids = list(DIFFICULTY_WEIGHTS.keys())
    difficulty_probs = list(DIFFICULTY_WEIGHTS.values())

    for i, prompt_text in enumerate(prompts):
        difficulty = random.choices(difficulty_ids, weights=difficulty_probs, k=1)[0]
        meta = DIFFICULTY_REGISTRY[difficulty]
        seed = random.randint(0, 1_000_000)

        log.info("Episode %d | difficulty=%s | seed=%d", i + 1, difficulty, seed)

        episode = rollout_once(
            trainer=trainer,
            tokenizer=tokenizer,
            dataset_prompt=prompt_text,
            system_prompt=system_prompt,
            max_turns=meta.max_steps,
            difficulty=difficulty,
            seed=seed,
        )

        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        evacuated_list.append(episode["evacuated"])
        final_health_list.append(episode["final_health"])
        step_reward_means.append(episode["mean_step_reward"])
        steps_taken_list.append(episode["steps_taken"])
        optimal_steps_list.append(episode["optimal_steps"])
        format_rates.append(episode["format_rate"])

        log.info(
            "  → %s | health=%.1f | mean_step=%.3f | steps=%d | fmt=%.0f%%",
            "EVACUATED" if episode["evacuated"] else "failed",
            episode["final_health"],
            episode["mean_step_reward"],
            episode["steps_taken"],
            episode["format_rate"] * 100,
        )

    return {
        "prompt_ids":       episode_prompt_ids,
        "completion_ids":   episode_completion_ids,
        "logprobs":         episode_logprobs,
        "evacuated":        evacuated_list,
        "final_health":     final_health_list,
        "mean_step_reward": step_reward_means,
        "steps_taken":      steps_taken_list,
        "optimal_steps":    optimal_steps_list,
        "format_rate":      format_rates,
    }


print("Rollout functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward functions (notebook "cell-14" analogue — 4 signals)
# ─────────────────────────────────────────────────────────────────────────────

def reward_evacuated(completions, **kwargs):
    """Primary signal: 1.0 if agent reached an exit, 0.0 otherwise."""
    evacuated = kwargs.get("evacuated")
    return [float(e) for e in evacuated] if evacuated else [0.0] * len(completions)


def reward_step_efficiency(completions, **kwargs):
    """Mean per-step reward from the environment rubrics."""
    rewards = kwargs.get("mean_step_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_format(completions, **kwargs):
    """+0.10 bonus when the model uses <think> tags on ≥50% of steps."""
    rates = kwargs.get("format_rate") or [0.0] * len(completions)
    return [0.10 if float(r) >= 0.5 else 0.0 for r in rates]


def reward_efficiency(completions, **kwargs):
    """Step-efficiency bonus: +0.10 if evacuated within H* steps, decays beyond."""
    steps_list  = kwargs.get("steps_taken")   or [0]   * len(completions)
    optimal     = kwargs.get("optimal_steps") or [50]  * len(completions)
    evacuated   = kwargs.get("evacuated")     or [0]   * len(completions)
    rewards = []
    for steps, h_star, evac in zip(steps_list, optimal, evacuated):
        if not int(evac):
            rewards.append(0.0)
            continue
        steps, h_star = int(steps), int(h_star)
        if steps <= h_star:
            rewards.append(0.10)
        else:
            rewards.append(round(0.10 * (0.85 ** (steps - h_star)), 4))
    return rewards


print("Reward functions: evacuated, step_efficiency, format, efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI + main (notebook cells 6, 16, 18, 20, 22, 25 inline)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training for Pyre (easy difficulty), OpenEnv-notebook style",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-id",     default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--output-dir",   default="./outputs/grpo_pyre_easy")
    p.add_argument("--push-to-hub",  action="store_true")
    p.add_argument("--debug",        action="store_true",
                   help="Log full prompt and completion text for each step")
    p.add_argument("--report-to",    default="trackio",
                   choices=["trackio", "tensorboard", "none"])
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled — full prompt/completion will be printed each step.")

    # ── Init tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Model: %s", args.model_id)

    # ── Create dataset ──────────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "prompt": ["Navigate the burning building and evacuate safely."] * args.dataset_size
    })
    log.info("Dataset: %d prompts", len(dataset))

    # ── Configure GRPO ──────────────────────────────────────────────────────
    #
    # Matching the reference notebook, with one Pyre-specific override:
    #
    #   max_completion_length  512 → 1024   (safe margin for Qwen3 think + JSON action)
    #
    # Everything else — lr, grad_accum, num_generations, vLLM colocate,
    # gradient checkpointing, push_to_hub — is identical to the reference.
    grpo_config = GRPOConfig(
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "eager",
        },
        num_train_epochs=1,
        learning_rate=5e-6,
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        warmup_steps=20,
        num_generations=2,
        max_completion_length=1024,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=8192,
        output_dir=args.output_dir,
        report_to=args.report_to,
        trackio_space_id=args.output_dir if args.report_to == "trackio" else None,
        logging_steps=1,
        save_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
    )
    log.info("Output: %s | vLLM mode: colocate", args.output_dir)

    # ── Create trainer ──────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_evacuated,
            reward_step_efficiency,
            reward_format,
            reward_efficiency,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    # GPU snapshot
    if torch.cuda.is_available():
        gpu       = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        total     = round(gpu.total_memory / 1024**3, 3)
        log.info("GPU: %s — %s GB total, %s GB reserved", gpu.name, total, start_mem)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        log.info(
            "Training time: %.1f min | Peak memory: %s GB",
            trainer_stats.metrics["train_runtime"] / 60, used,
        )

    # ── Save and (optionally) push ──────────────────────────────────────────
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()
        log.info("Model saved to %s and pushed to Hub.", args.output_dir)
    else:
        log.info("Model saved to %s.", args.output_dir)
