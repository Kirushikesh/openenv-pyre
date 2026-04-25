"""
pyre_evals.py — Proprietary-LLM baseline evaluation for the Pyre fire-evacuation environment.

Tests how well a top LLM (GPT-5.4-nano, Claude 3.5, etc.) performs at navigating a burning building
to an exit under partial observability, time pressure, and a spreading-fire hazard.

Because the model has never been trained on this task, its behaviour acts as a clean
diagnostic signal:
  • High evacuated rate  → the environment reward signal, observations, and action space are
                           legible to a capable model.
  • Systematic failures  → reveal bugs or design flaws (e.g. narrative is ambiguous, exits
                           are unreachable, action parsing is fragile, damage is too fast, etc.).
  • Low think_rate       → the prompt isn't eliciting step-by-step reasoning.
  • Low parse_rate       → the action schema is hard to follow.

Metrics per episode
───────────────────
  evacuated        — 1 if agent reached a safe exit, 0 otherwise          (primary signal)
  cause_of_end     — "evacuated" | "death" | "timeout"
  final_health     — agent health at episode end (0–100)
  total_reward     — cumulative step + episode-end reward
  mean_step_reward — total_reward / steps_taken
  steps_taken      — steps used (compare to max_steps for the difficulty)
  think_rate       — fraction of steps where model emitted <think>…</think>
  parse_rate       — fraction of steps where a valid action was extracted
  format_score_avg — mean format quality (0.0–1.0) across all steps

Usage
─────
  # 1. Add keys to pyre_env/.env (never commit this file):
  #
  #      OPENAI_API_KEY=sk-...
  #      ANTHROPIC_API_KEY=sk-ant-...
  #
  # 2. Start the Pyre server (from pyre_env/):
  #      uv run server
  #
  # 3. Run the evaluator (from pyre_env/):
  python evals.py --env-url http://localhost:8000

  # Multiple difficulties, more seeds:
  python pyre_evals.py --difficulties easy medium hard --seeds 5

  # Filter to a single difficulty, verbose logging:
  python pyre_evals.py --difficulties medium --seeds 3 --verbose

  # Custom model and temperature:
  python pyre_evals.py --model gpt-5.4-nano --temperature 0.0

  # Anthropic provider:
  python pyre_evals.py --provider anthropic --model claude-3-5-sonnet-20241022

  # RITS endpoint:
  python pyre_evals.py --provider rits --model meta-llama/llama-3-70b-instruct

  # Save per-episode LLM traces for debugging:
  python pyre_evals.py --debug --verbose

Dependencies
────────────
  pip install langchain-openai langchain-anthropic requests python-dotenv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pyre_env import PyreEnv, PyreAction

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit(
        "python-dotenv is required. Install it with:  pip install python-dotenv"
    )

# Load .env from the same directory as this script (pyre_env/.env).
# All API keys and endpoint URLs must be defined there — do NOT export them
# in the shell or hard-code them here.
load_dotenv()

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pyre_evals")


# ═══════════════════════════════════════════════════════════════════════════════
# Difficulty registry
# ═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY_CONFIGS: List[Dict[str, Any]] = [
    {
        "difficulty": "easy",
        "max_steps": 200,
        "description": "1 fire source · slow spread · calm wind · high humidity",
    },
    {
        "difficulty": "medium",
        "max_steps": 150,
        "description": "2–4 fire sources · moderate spread · any wind · moderate humidity",
    },
    {
        "difficulty": "hard",
        "max_steps": 100,
        "description": "3–5 fire sources · fast spread · always windy · low humidity",
    },
]

DIFFICULTY_MAP: Dict[str, Dict[str, Any]] = {d["difficulty"]: d for d in DIFFICULTY_CONFIGS}


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
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

REWARD SIGNAL (shown in history after each step)
- Positive reward  → you moved closer to an exit or played a smart move.
- Negative reward  → you moved into danger, stalled, or wasted a step.
- Use the reward trend to judge if your current direction is working.

STRATEGY TIPS
- Use `look` to scout a direction before entering an unknown corridor.
- Closing a door between you and fire buys time; re-open when clear.
- Prefer moves that increase reward — progress toward the exit is rewarded.
- If smoke is heavy, back away; your health drains fast in thick smoke.
- Door IDs (e.g. door_3) appear in the Visible objects list — use them with the door action.
""").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_message(obs: Dict[str, Any], history: List[str]) -> str:
    """Convert a raw observation dict + history into the LLM user message."""

    narrative = obs.get("narrative", "(no narrative)")
    # Strip the "Available actions:" line the narrative builder appends — the
    # system prompt already documents the full action schema.
    narrative = re.sub(r"\nAvailable actions:.*$", "", narrative, flags=re.MULTILINE)

    # ── Structured status line ──────────────────────────────────────────────
    health       = obs.get("agent_health", "?")
    health_st    = obs.get("health_status", "?")
    location     = obs.get("location_label", "?")
    smoke        = obs.get("smoke_level", "none")
    fire_vis     = obs.get("fire_visible", False)
    fire_dir     = obs.get("fire_direction") or "none"
    wind         = obs.get("wind_dir", "CALM")
    elapsed      = obs.get("elapsed_steps", 0)
    last_fb      = obs.get("last_action_feedback", "")
    blocked_exits = obs.get("blocked_exit_ids", [])
    visible_objs = obs.get("visible_objects", [])
    audible      = obs.get("audible_signals", [])

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

    # ── History ─────────────────────────────────────────────────────────────
    history_str = ""
    if history:
        recent = history[-8:]  # last 8 steps to stay within context limits
        history_str = "=== RECENT ACTION HISTORY (action → feedback → reward → health) ===\n" + "\n".join(recent) + "\n\n"

    return (
        f"=== CURRENT OBSERVATION ===\n{narrative}\n\n"
        f"=== STATUS ===\n{status_line}\n\n"
        + history_str
        + "What is your next action? Respond with <think>...</think> then a single JSON action."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Action parser
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_ACTIONS = {"move", "door", "look", "wait"}
_VALID_DIRECTIONS = {"north", "south", "east", "west"}
_VALID_DOOR_STATES = {"open", "close"}

_FALLBACK_ACTION = {"action": "wait"}


def _validate_pyre_action(blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a sanitised action dict or None if the blob is unusable."""
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
        out["target_id"] = str(tid)
        out["door_state"] = ds

    # "wait" needs no extra fields

    return out


def _parse_pyre_action(text: str) -> Tuple[Dict[str, Any], float]:
    """Extract a Pyre action from raw LLM text.

    Returns (action_dict, format_score) where format_score reflects output quality:
      1.0  — valid JSON + <think> tags
      0.7  — valid JSON, no <think>
      0.4  — partial JSON rescued via regex
      0.1  — action keyword found in raw text (last resort)
      0.0  — completely unparseable → {"action": "wait"} fallback
    """
    has_think = "<think>" in text and "</think>" in text

    # ── Level 1: well-formed JSON ────────────────────────────────────────────
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

    # ── Level 2: regex — find the innermost {...} with "action" key ──────────
    for m in re.finditer(r'\{[^{}]+\}', text):
        try:
            blob = json.loads(m.group())
            if isinstance(blob, dict) and "action" in blob:
                action = _validate_pyre_action(blob)
                if action is not None:
                    return action, 0.4
        except json.JSONDecodeError:
            continue

    # ── Level 3: bare keyword extraction ────────────────────────────────────
    lower = text.lower()
    # move <direction>
    for d in _VALID_DIRECTIONS:
        if f"move {d}" in lower or f'move.*{d}' in lower:
            return {"action": "move", "direction": d}, 0.1
    # look <direction>
    for d in _VALID_DIRECTIONS:
        if f"look {d}" in lower:
            return {"action": "look", "direction": d}, 0.1
    # door open/close target_id
    door_m = re.search(r'door[_\s]*([\w]+)', lower)
    if door_m:
        tid = door_m.group(1)
        ds  = "close" if "clos" in lower else "open"
        return {"action": "door", "target_id": tid, "door_state": ds}, 0.1
    # wait
    if "wait" in lower:
        return {"action": "wait"}, 0.1

    # ── Level 4: total parse failure ─────────────────────────────────────────
    return dict(_FALLBACK_ACTION), 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LLM factory
# ═══════════════════════════════════════════════════════════════════════════════

def _require_env(key: str, hint: str) -> str:
    """Return the value of an env var or exit with a helpful message."""
    val = os.getenv(key, "")
    if not val:
        raise SystemExit(
            f"Missing required environment variable: {key}\n"
            f"Add it to your .env file:  {key}={hint}"
        )
    return val


def _build_llm(provider: str, model: str, temperature: float):
    """Construct a LangChain chat model; credentials come from .env via load_dotenv()."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = _require_env("OPENAI_API_KEY", "sk-...")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=3,
            api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = _require_env("ANTHROPIC_API_KEY", "sk-ant-...")
        return ChatAnthropic(  # type: ignore[attr-defined]
            model=model,
            temperature=temperature,
            max_retries=3,
            api_key=api_key,
        )

    if provider == "rits":
        import httpx
        from langchain_openai import ChatOpenAI
        base_url = _require_env("RITS_URL", "https://inference.example.com/v1")
        rits_key = _require_env("RITS_API_KEY", "<your-rits-key>")
        # trust_env=False bypasses macOS system-proxy and any HTTPS_PROXY env vars
        # so requests go directly to the RITS endpoint.
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=2,
            api_key="/",
            base_url=base_url,
            default_headers={"RITS_API_KEY": rits_key},
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )

    raise SystemExit(f"Unknown provider '{provider}'. Choose from: openai, anthropic, rits")


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(
    llm,
    env_url: str,
    seed: int,
    difficulty: str,
    max_steps: int,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one full episode using the PyreEnv sync client (WebSocket-based, stateful).

    The episode is self-contained: reset → loop(observe → LLM → act) → done.
    Uses PyreEnv.sync() so the WebSocket session persists across all steps of
    the episode — the HTTP /step endpoint is stateless and cannot be used for
    multi-step rollouts.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    step_rewards:      List[float] = []
    history:           List[str]   = []
    llm_responses_log: List[str]   = []
    steps_taken    = 0
    think_steps    = 0
    parsed_steps   = 0
    fmt_scores:    List[float] = []
    cause_of_end   = "timeout"
    final_health   = 0.0
    agent_evacuated = False

    try:
        with PyreEnv(base_url=env_url).sync() as env:
            # ── Reset ─────────────────────────────────────────────────────────
            result = env.reset(difficulty=difficulty, seed=seed)
            obs    = result.observation   # PyreObservation
            done   = result.done

            # Optionally save initial state snapshot for debugging
            if debug_dir is not None:
                try:
                    state_resp = requests.get(f"{env_url}/state", timeout=10)
                    if state_resp.ok:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        (debug_dir / f"{difficulty}_seed{seed}_init_state.json").write_text(
                            json.dumps(state_resp.json(), indent=2)
                        )
                except Exception as exc:
                    log.warning("Could not save initial state: %s", exc)

            # ── Episode loop ─────────────────────────────────────────────────
            for _step in range(max_steps):
                if done:
                    break

                # Convert PyreObservation to dict for the prompt builder
                obs_dict = obs.model_dump()

                user_msg = _build_user_message(obs_dict, history)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]

                # ── LLM call ─────────────────────────────────────────────────
                try:
                    response = llm.invoke(messages)
                    completion_text = response.content
                except Exception as exc:
                    log.warning("LLM call failed at step %d: %s", _step + 1, exc)
                    step_rewards.append(-0.20)
                    break

                llm_responses_log.append(f"## Step {_step + 1}\n{completion_text}\n")

                has_think = "<think>" in completion_text and "</think>" in completion_text
                if has_think:
                    think_steps += 1

                steps_taken += 1

                # ── Parse action ─────────────────────────────────────────────
                action_dict, fmt_score = _parse_pyre_action(completion_text)
                fmt_scores.append(fmt_score)

                if fmt_score > 0.0:
                    parsed_steps += 1
                else:
                    log.debug("  step=%d  UNPARSEABLE — using wait fallback", _step + 1)

                log.debug(
                    "  step=%d  fmt=%.1f  action=%s",
                    _step + 1, fmt_score, json.dumps(action_dict),
                )

                # ── Step the environment via PyreEnv client ───────────────────
                try:
                    result      = env.step(PyreAction(**action_dict))
                    obs         = result.observation
                    step_reward = float(result.reward or 0.0)
                    done        = result.done
                except Exception as exc:
                    log.warning("Step failed at step %d: %s", _step + 1, exc)
                    step_rewards.append(-0.20)
                    break

                # Format penalty: imperfect parse loses up to -0.10
                fmt_penalty = (1.0 - fmt_score) * -0.10
                step_rewards.append(step_reward + fmt_penalty)

                feedback = obs.last_action_feedback or ""
                history.append(
                    f"Step {_step + 1}: {json.dumps(action_dict)}"
                    + (f"\n  → {feedback}" if feedback else "")
                    + f"\n  reward: {step_reward:+.3f}  health: {obs.agent_health:.1f}"
                )

            # ── Final state ───────────────────────────────────────────────────
            agent_evacuated = obs.agent_evacuated
            final_health    = float(obs.agent_health)

    except Exception as exc:
        log.error("Episode failed (difficulty=%s seed=%d): %s", difficulty, seed, exc)
        return {
            "difficulty": difficulty, "seed": seed,
            "error": str(exc), "evacuated": 0,
        }

    # ── Determine cause of end ────────────────────────────────────────────────
    if agent_evacuated:
        cause_of_end = "evacuated"
    elif final_health <= 0.0:
        cause_of_end = "death"
    else:
        cause_of_end = "timeout"

    # ── Optionally dump LLM trace ─────────────────────────────────────────────
    if debug_dir is not None and llm_responses_log:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"{difficulty}_seed{seed}_llm_trace.md").write_text(
                "\n".join(llm_responses_log)
            )
        except Exception as exc:
            log.warning("Could not save LLM trace: %s", exc)

    total_reward     = sum(step_rewards)
    mean_step_reward = total_reward / max(len(step_rewards), 1)
    think_rate       = think_steps / max(steps_taken, 1)
    parse_rate       = parsed_steps / max(steps_taken, 1)
    fmt_score_avg    = sum(fmt_scores) / max(len(fmt_scores), 1)

    return {
        "difficulty":        difficulty,
        "seed":              seed,
        "evacuated":         int(agent_evacuated),
        "cause_of_end":      cause_of_end,
        "final_health":      round(final_health, 2),
        "total_reward":      round(total_reward, 4),
        "mean_step_reward":  round(mean_step_reward, 4),
        "steps_taken":       steps_taken,
        "max_steps":         max_steps,
        "think_rate":        round(think_rate, 4),
        "parse_rate":        round(parse_rate, 4),
        "format_score_avg":  round(fmt_score_avg, 4),
        "error":             None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _pct(vals: List[float]) -> str:
    return f"{_avg(vals) * 100:.0f}%"


def print_summary(results: List[Dict[str, Any]], model_name: str) -> None:
    """Print a per-difficulty summary table and a short diagnosis section."""
    by_diff: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        if not r.get("error"):
            by_diff[r["difficulty"]].append(r)

    # ── Header ───────────────────────────────────────────────────────────────
    col = "{:<10} {:>8} {:>10} {:>10} {:>11} {:>9} {:>9} {:>9}"
    header = col.format(
        "Difficulty", "Evac%", "AvgHealth", "TotalRew",
        "MeanStpRew", "Steps/Max", "Think%", "Parse%",
    )
    sep = "=" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  PYRE EVAL — model: {model_name}")
    print(sep)
    print(header)
    print("-" * len(header))

    diagnosis: List[str] = []

    for cfg in DIFFICULTY_CONFIGS:
        diff  = cfg["difficulty"]
        rows  = by_diff.get(diff, [])
        if not rows:
            print(col.format(diff, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"))
            continue

        evac_rate        = _avg([r["evacuated"]         for r in rows])
        avg_health       = _avg([r["final_health"]       for r in rows])
        avg_total_rew    = _avg([r["total_reward"]        for r in rows])
        avg_step_rew     = _avg([r["mean_step_reward"]    for r in rows])
        avg_steps        = _avg([r["steps_taken"]         for r in rows])
        avg_think        = _avg([r["think_rate"]          for r in rows])
        avg_parse        = _avg([r["parse_rate"]          for r in rows])
        max_steps        = rows[0]["max_steps"]

        cause_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            cause_counts[r["cause_of_end"]] += 1

        print(col.format(
            diff,
            f"{evac_rate * 100:.0f}%",
            f"{avg_health:.1f}",
            f"{avg_total_rew:+.2f}",
            f"{avg_step_rew:+.3f}",
            f"{avg_steps:.0f}/{max_steps}",
            f"{avg_think * 100:.0f}%",
            f"{avg_parse * 100:.0f}%",
        ))

        # Accumulate diagnosis hints
        death_n    = cause_counts.get("death", 0)
        timeout_n  = cause_counts.get("timeout", 0)
        evac_n     = cause_counts.get("evacuated", 0)
        n          = len(rows)

        if avg_parse < 0.80:
            diagnosis.append(
                f"[{diff}] Low parse rate ({avg_parse*100:.0f}%) — "
                "action schema may be unclear; consider simplifying JSON keys."
            )
        if avg_think < 0.50:
            diagnosis.append(
                f"[{diff}] Low think rate ({avg_think*100:.0f}%) — "
                "model is skipping <think> tags; prompt may need stronger CoT instruction."
            )
        if evac_n == 0 and diff == "easy":
            diagnosis.append(
                "[easy] Zero evacuations on easy difficulty — "
                "check exit reachability, narrative clarity, or BFS distance from spawn."
            )
        if death_n / n > 0.80:
            diagnosis.append(
                f"[{diff}] {death_n}/{n} episodes end in death — "
                "damage rates may be too high or smoke/fire proximity warnings are not clear."
            )
        if timeout_n / n > 0.80:
            diagnosis.append(
                f"[{diff}] {timeout_n}/{n} episodes time out — "
                "model may be looping/waiting; exits might be hard to locate from narratives."
            )
        if avg_step_rew < -0.3 and evac_rate < 0.1:
            diagnosis.append(
                f"[{diff}] Very negative mean step reward ({avg_step_rew:+.3f}) with near-zero "
                "success — model may be actively moving into fire; check DangerPenalty trigger conditions."
            )

    print(sep + "\n")

    # ── Diagnosis section ────────────────────────────────────────────────────
    if diagnosis:
        print("DIAGNOSTICS (environment design signals)")
        print("-" * 50)
        for d in diagnosis:
            print(f"  • {d}")
        print()
    else:
        print("  No red-flag patterns detected — environment appears legible to this model.\n")


def save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "difficulty", "seed", "evacuated", "cause_of_end",
        "final_health", "total_reward", "mean_step_reward",
        "steps_taken", "max_steps",
        "think_rate", "parse_rate", "format_score_avg", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    log.info("Results saved → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proprietary-LLM baseline eval on the Pyre fire-evacuation environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-url", default="http://localhost:8000",
        help="Pyre server base URL",
    )
    parser.add_argument(
        "--provider", default="openai",
        choices=["openai", "anthropic", "rits"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model", default="gpt-5.4-nano",
        help="Model name / slug for the chosen provider",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = deterministic greedy)",
    )
    parser.add_argument(
        "--difficulties", nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to evaluate",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds per difficulty level",
    )
    parser.add_argument(
        "--seed-start", type=int, default=1,
        help="Starting seed value (seeds = seed_start, seed_start+1, ...)",
    )
    parser.add_argument(
        "--output-dir", default="./outputs/pyre_evals",
        help="Directory for CSV results and debug traces",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save per-episode LLM traces and initial states to --output-dir/debug/",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show DEBUG-level per-step logs",
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # ── Filter difficulties ───────────────────────────────────────────────────
    diffs_to_run = [d for d in DIFFICULTY_CONFIGS if d["difficulty"] in args.difficulties]
    if not diffs_to_run:
        raise SystemExit(
            f"No matching difficulties. Available: {[d['difficulty'] for d in DIFFICULTY_CONFIGS]}"
        )

    log.info("Model       : %s (provider=%s, temperature=%.1f)", args.model, args.provider, args.temperature)
    log.info("Env URL     : %s", args.env_url)
    log.info("Difficulties: %s", [d["difficulty"] for d in diffs_to_run])
    log.info("Seeds       : %d (%d–%d)", args.seeds, args.seed_start, args.seed_start + args.seeds - 1)

    # ── Health check ─────────────────────────────────────────────────────────
    try:
        health = requests.get(f"{args.env_url}/health", timeout=5)
        health.raise_for_status()
        log.info("Server      : healthy ✓")
    except Exception as exc:
        raise SystemExit(f"Server not reachable at {args.env_url}: {exc}")

    # ── Build LLM ────────────────────────────────────────────────────────────
    llm = _build_llm(args.provider, args.model, args.temperature)

    # ── Run episodes ─────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    seeds      = list(range(args.seed_start, args.seed_start + args.seeds))
    total      = len(diffs_to_run) * len(seeds)
    done_count = 0
    debug_dir  = Path(args.output_dir) / "debug" if args.debug else None

    for diff_cfg in diffs_to_run:
        difficulty = diff_cfg["difficulty"]
        max_steps  = diff_cfg["max_steps"]

        for seed in seeds:
            done_count += 1
            log.info(
                "[%d/%d] difficulty=%-8s seed=%d",
                done_count, total, difficulty, seed,
            )

            result = run_episode(
                llm=llm,
                env_url=args.env_url,
                seed=seed,
                difficulty=difficulty,
                max_steps=max_steps,
                debug_dir=debug_dir,
            )
            results.append(result)

            if result.get("error"):
                log.warning("  ERROR: %s", result["error"])
            else:
                evac_sym = "✓ EVACUATED" if result["evacuated"] else f"✗ {result['cause_of_end'].upper()}"
                log.info(
                    "  %-14s  health=%5.1f  reward=%+.2f  steps=%d/%d  think=%d%%  parse=%d%%",
                    evac_sym,
                    result["final_health"],
                    result["total_reward"],
                    result["steps_taken"],
                    max_steps,
                    result["think_rate"] * 100,
                    result["parse_rate"] * 100,
                )

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary(results, model_name=f"{args.provider}/{args.model}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    csv_path   = Path(args.output_dir) / f"pyre_{model_slug}_{timestamp}.csv"
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
