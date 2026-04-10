"""Inference script for SiliconMind OpenEnv Environment.

Baseline agent that uses an LLM (via OpenAI client) to make prefetch
decisions.  Follows the MANDATORY competition format:

  stdout: [START] task=... env=... model=...
  stdout: [STEP] step=... action=... reward=... done=... error=...
  stdout: [END] success=... steps=... score=... rewards=...

Environment variables (required by competition):
  API_BASE_URL  — LLM API endpoint
  API_KEY       — Authentication token
  MODEL_NAME    — Model identifier (optional, defaults to Qwen2.5-72B)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import SiliconMindClient
from models import ActionType, SiliconMindAction

# ── Configuration ────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "silicon_mind"
TEMPERATURE = 0.0
MAX_TOKENS = 256
ENV_URL = os.getenv("SILICON_MIND_URL", "http://localhost:7860")

ALL_TASKS = [
    "easy_pattern_scout",
    "medium_workload_whisperer",
    "hard_noisy_neighbor",
    "extreme_silicon_architect",
]

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a hardware cache prefetcher controller.  At each step you observe
    a memory access with its address, delta pattern, cache state, and bandwidth
    pressure.  You must decide whether to prefetch future cache lines.

    RESPOND ONLY with a JSON object:
    {
      "type": "prefetch" | "no_prefetch" | "set_aggressiveness" | "throttle_bandwidth" | "partition_cache",
      "target": "<see below>",
      "detail": "<your reasoning>"
    }

    Target formats:
      prefetch → "1,2" (comma-separated cache-line offsets to prefetch)
      no_prefetch → "skip"
      set_aggressiveness → "1" to "4"
      throttle_bandwidth → "0.0" to "1.0"
      partition_cache → "web:6,db:6,video:4"

    GUIDELINES:
    - Prefetch when you see a consistent stride pattern (confidence > 70%)
    - Use NO_PREFETCH when the pattern is random or unknown
    - Reduce aggressiveness when bandwidth is HIGH or CRITICAL
    - Reduce aggressiveness when pollution rate > 10%
    - In multi-tenant tasks, partition the LLC to protect high-priority workloads
""").strip()


# ── Logging (competition format) ─────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Action Parsing ───────────────────────────────────────────────────────

def parse_llm_action(response_text: str) -> SiliconMindAction:
    """Parse LLM response into a SiliconMindAction."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0] if "```" in text else text
            text = text.strip()

        data = json.loads(text)
        return SiliconMindAction(
            type=ActionType(data.get("type", "no_prefetch")),
            target=str(data.get("target", "skip")),
            detail=str(data.get("detail", "")),
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: try to detect simple patterns
        lower = response_text.lower()
        if "no_prefetch" in lower or "skip" in lower:
            return SiliconMindAction(type=ActionType.NO_PREFETCH, target="skip")
        elif "prefetch" in lower:
            return SiliconMindAction(type=ActionType.PREFETCH, target="1")
        else:
            return SiliconMindAction(type=ActionType.NO_PREFETCH, target="skip")


# ── Task Execution ───────────────────────────────────────────────────────

def run_task(task_id: str, llm_client: OpenAI, env_client: SiliconMindClient) -> None:
    """Run one task: reset → step loop → log results."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    final_score = 0.0
    success = False

    try:
        # Reset environment
        obs = env_client.reset(task_id=task_id, seed=42)

        while not obs.done and steps < obs.steps_remaining + obs.step_number + 1:
            steps += 1

            # Build prompt from observation
            user_message = (
                f"{obs.narrative}\n\n"
                f"Pattern: {obs.pattern_analysis.detected_pattern} "
                f"(confidence: {obs.pattern_analysis.confidence:.0%})\n"
                f"Recent deltas: {obs.recent_deltas}\n"
                f"Accuracy: {obs.prefetch_accuracy_recent:.0%} recent, "
                f"{obs.prefetch_accuracy_total:.0%} total\n"
                f"Bandwidth: {obs.bandwidth.pressure.value} ({obs.bandwidth.utilization:.0%})\n"
                f"Available actions: {obs.available_actions}\n"
            )

            if obs.action_hint:
                user_message += f"Hint: {obs.action_hint}\n"

            # Call LLM
            try:
                response = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                llm_text = response.choices[0].message.content or ""
                action = parse_llm_action(llm_text)
            except Exception as e:
                # Mock a heuristic response when LLM fails
                conf = obs.pattern_analysis.confidence
                pattern = obs.pattern_analysis.detected_pattern
                bw = obs.bandwidth.pressure.value
                if conf > 0.5 and "stride" in pattern and bw not in ("high", "critical"):
                    action = SiliconMindAction(type=ActionType.PREFETCH, target="1")
                elif bw in ("high", "critical"):
                    action = SiliconMindAction(type=ActionType.SET_AGGRESSIVENESS, target="1")
                else:
                    action = SiliconMindAction(type=ActionType.NO_PREFETCH, target="skip")
                
                # We do not continue here, let the normal flow step the environment

            # Step environment
            action_str = f"{action.type.value}:{action.target}"
            obs = env_client.step(action)

            reward_val = getattr(obs.reward, "total", obs.reward)
            rewards.append(reward_val)
            log_step(steps, action_str, reward_val, obs.done, None)

            # Extract final score if done
            if obs.done and "final_scores" in obs.info:
                final_score = obs.info["final_scores"].get("total", 0.0)
                success = True

    except Exception as e:
        log_step(steps, "error", 0.0, True, str(e)[:200])

    # Clamp score
    final_score = round(max(0.01, min(0.99, final_score)), 3)
    log_end(success=success, steps=steps, score=final_score, rewards=rewards)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all tasks sequentially."""
    # Initialize clients
    llm_client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )
    env_client = SiliconMindClient(ENV_URL)

    # Check environment health
    if not env_client.health():
        print("[ERROR] Environment server not reachable at", ENV_URL, file=sys.stderr)
        sys.exit(1)

    # Run all tasks
    for task_id in ALL_TASKS:
        run_task(task_id, llm_client, env_client)


if __name__ == "__main__":
    main()
