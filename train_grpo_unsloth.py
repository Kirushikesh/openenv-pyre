"""
train_grpo_unsloth.py — GRPO + LoRA training for Pyre via Unsloth
                         (PyreEnvironment in-process, multi-turn episode rollout)

Mirrors train_grpo_openenv.py but replaces the TRL-native stack with:
  • FastLanguageModel  — memory-efficient model loading + LoRA
  • model.fast_generate — vLLM-backed generation inside the episode rollout
  • adamw_8bit / cosine schedule — Unsloth-recommended training config
  • model.save_lora / save_pretrained_merged — Unsloth save API

Everything else (environment, prompts, reward functions, rollout structure) is
identical to the TRL version so the two scripts stay directly comparable.

Usage
─────
  python train_grpo_unsloth.py
  python train_grpo_unsloth.py --model-id unsloth/Qwen3-1.7B --dataset-size 1000
  python train_grpo_unsloth.py --model-id unsloth/Qwen3-4B --lora-rank 32 --save-merged

Dependencies
────────────
  uv sync --extra train-unsloth
  (installs unsloth, vllm, trl, datasets, etc.)
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
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
log = logging.getLogger("grpo_unsloth")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Difficulty registry + stratified mix
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DifficultyMeta:
    difficulty: str
    max_steps: int
    optimal_steps: int


DIFFICULTY_REGISTRY: Dict[str, DifficultyMeta] = {
    "easy": DifficultyMeta("easy", max_steps=200, optimal_steps=50),
}

DIFFICULTY_WEIGHTS: Dict[str, float] = {
    "easy": 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. System prompt
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
# 3. Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_user_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    """Build the user turn from the current PyreObservation dict + action history."""
    narrative = obs.get("narrative", "(no narrative)")
    narrative = re.sub(r"\nAvailable actions:.*$", "", narrative, flags=re.MULTILINE)

    health        = obs.get("agent_health", 0.0)
    health_st     = obs.get("health_status", "?")
    location      = obs.get("location_label", "?")
    smoke         = obs.get("smoke_level", "none")
    fire_vis      = obs.get("fire_visible", False)
    fire_dir      = obs.get("fire_direction") or "none"
    wind          = obs.get("wind_dir", "CALM")
    elapsed       = obs.get("elapsed_steps", 0)
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Module-level generation objects
#    Populated in __main__ once the tokenizer eos_token is known.
# ─────────────────────────────────────────────────────────────────────────────

# model and tokenizer are set as module-level globals in __main__ so that
# rollout_once (called inside rollout_func, which is called by GRPOTrainer)
# can reach them without threading them through every call frame.
model = None
tokenizer = None

VLLM_SAMPLING_PARAMS: Optional[SamplingParams] = None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Rollout function
# ─────────────────────────────────────────────────────────────────────────────

def rollout_once(
    dataset_prompt: str,
    max_turns: int,
    difficulty: str,
    seed: int,
) -> Dict[str, Any]:
    """Execute one full Pyre episode using Unsloth's model.fast_generate.

    Generation is handled by model.fast_generate (Unsloth vLLM backend).
    Token IDs are extracted directly from the vLLM RequestOutput so that
    GRPOTrainer can compute the GRPO loss from them.
    """
    prompt_ids:     List[int]   = []
    completion_ids: List[int]   = []
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

            obs_dict    = obs.model_dump()
            user_prompt = make_user_prompt(obs_dict, history)
            messages    = [
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

            # ── Unsloth vLLM generation ─────────────────────────────────────
            vllm_out        = model.fast_generate([prompt_text], sampling_params=VLLM_SAMPLING_PARAMS)[0]
            completion_text = vllm_out.outputs[0].text

            # Accumulate token IDs for GRPO loss computation
            prompt_ids.extend(list(vllm_out.prompt_token_ids))
            completion_ids.extend(list(vllm_out.outputs[0].token_ids))
            # ────────────────────────────────────────────────────────────────

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
        "logprobs":         [],       # fast_generate does not expose per-token logprobs
        "evacuated":        int(agent_evacuated),
        "final_health":     round(final_health, 2),
        "mean_step_reward": round(mean_step_reward, 4),
        "steps_taken":      steps_taken,
        "optimal_steps":    meta.optimal_steps,
        "format_rate":      round(format_rate, 4),
    }


def rollout_func(prompts, trainer=None):  # trainer kept for API compatibility, unused
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

    difficulty_ids   = list(DIFFICULTY_WEIGHTS.keys())
    difficulty_probs = list(DIFFICULTY_WEIGHTS.values())

    for i, prompt_text in enumerate(prompts):
        difficulty = random.choices(difficulty_ids, weights=difficulty_probs, k=1)[0]
        meta       = DIFFICULTY_REGISTRY[difficulty]
        seed       = random.randint(0, 1_000_000)

        log.info("Episode %d | difficulty=%s | seed=%d", i + 1, difficulty, seed)

        episode = rollout_once(
            dataset_prompt=prompt_text,
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reward functions  (identical to TRL version)
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
    steps_list = kwargs.get("steps_taken")   or [0]  * len(completions)
    optimal    = kwargs.get("optimal_steps") or [50] * len(completions)
    evacuated  = kwargs.get("evacuated")     or [0]  * len(completions)
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


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO + LoRA training for Pyre via Unsloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model-id",               default="unsloth/Qwen3-1.7B")
    p.add_argument("--lora-rank",              type=int,   default=32,
                   help="LoRA rank r; lora_alpha is set to r*2")
    p.add_argument("--load-in-4bit",           action="store_true",
                   help="Load model in 4-bit (QLoRA). Default: 16-bit LoRA")
    p.add_argument("--max-seq-length",         type=int,   default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                   help="Fraction of GPU memory reserved for vLLM")
    # Training
    p.add_argument("--dataset-size",           type=int,   default=1000)
    p.add_argument("--output-dir",             default="./outputs/grpo_pyre_unsloth")
    p.add_argument("--report-to",              default="tensorboard",
                   choices=["trackio", "tensorboard", "none"])
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--debug",                  action="store_true",
                   help="Log full prompt and completion text for each step")
    # Save / publish
    p.add_argument("--save-merged",            action="store_true",
                   help="After training, also save a merged 16-bit checkpoint")
    p.add_argument("--push-to-hub",            action="store_true")
    p.add_argument("--hub-repo",               default=None,
                   help="HF repo id for push_to_hub_merged, e.g. 'user/model'")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load model + tokenizer via Unsloth ──────────────────────────────────
    log.info("Loading model '%s' via FastLanguageModel …", args.model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,                     # activates Unsloth vLLM backend
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    log.info("Applying LoRA (r=%d, alpha=%d) …", args.lora_rank, args.lora_rank * 2)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",    # Unsloth memory-efficient GC
        random_state=args.seed,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("Trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", trainable / total * 100)

    # ── Wire up module-level sampling params ────────────────────────────────
    # VLLM_SAMPLING_PARAMS, model, and tokenizer are declared as module-level
    # globals above.  Assigning here (we are at module scope, not inside a
    # function) updates those globals so rollout_once can reach them when
    # GRPOTrainer calls rollout_func during training.
    VLLM_SAMPLING_PARAMS = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        max_tokens=1024,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": "Navigate the burning building and evacuate safely."},
            ]
        ] * args.dataset_size
    })
    log.info("Dataset: %d prompts", len(dataset))

    # ── GRPOConfig — Unsloth style ──────────────────────────────────────────
    grpo_config = GRPOConfig(
        vllm_sampling_params=VLLM_SAMPLING_PARAMS,
        temperature=0.7,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_completion_length=1024,
        num_train_epochs=1,
        gradient_checkpointing=True,
        output_dir=args.output_dir,
        report_to=args.report_to,
        trackio_space_id=args.output_dir if args.report_to == "trackio" else None,
        logging_steps=1,
        save_steps=10,
        push_to_hub=False,             # push handled manually via Unsloth API below
        seed=args.seed,
    )
    log.info("Output: %s", args.output_dir)

    # ── GRPOTrainer ─────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
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

    if torch.cuda.is_available():
        gpu       = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        total_mem = round(gpu.total_memory / 1024**3, 3)
        log.info("GPU: %s — %s GB total, %s GB reserved", gpu.name, total_mem, start_mem)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        log.info(
            "Training time: %.1f min | Peak memory: %s GB",
            trainer_stats.metrics["train_runtime"] / 60, used,
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Always save LoRA adapters
    model.save_lora(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("LoRA adapters saved to %s", args.output_dir)

    # Optionally merge weights into a full 16-bit checkpoint
    if args.save_merged:
        merged_dir = args.output_dir + "_merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        log.info("Merged 16-bit checkpoint saved to %s", merged_dir)

    # Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        repo = args.hub_repo or args.output_dir
        model.push_to_hub_merged(
            repo,
            tokenizer,
            save_method="merged_16bit",
            token=os.getenv("HF_TOKEN"),
        )
        log.info("Model pushed to Hub: %s", repo)
