"""Per-step dense reward computation using Pythia-inspired 9-level classification.

This module classifies each agent action into one of 9 outcomes:
  RAT       (+20)  Accurate and timely prefetch
  RAL       (+12)  Accurate but late prefetch
  RESTRAINT (+8)   Correct NO_PREFETCH on unpredictable access
  REDUNDANT (-2)   Prefetched a line already in cache
  RNP_H     (-2)   No-prefetch under high bandwidth pressure (acceptable)
  RNP_L     (-4)   No-prefetch under low bandwidth (missed opportunity)
  RIN_L     (-8)   Inaccurate prefetch, low bandwidth (wrong but cheap)
  RIN_H     (-14)  Inaccurate prefetch, high bandwidth (wrong AND expensive)
  POLLUTION (-18)  Prefetch evicted a useful line

The raw reward is then normalized to [0.001, 0.999] for OpenEnv compliance.

Reference: Pythia (MICRO '21), Bera et al. — Table of reward levels.
"""

from __future__ import annotations

from models import (
    PrefetchOutcome,
    Reward,
    RewardComponents,
    REWARD_VALUES,
    normalize_reward,
)
from engine.counters import StepStats, EpisodeStats


def classify_prefetch_outcome(
    step: StepStats,
    bw_high: bool,
) -> PrefetchOutcome:
    """Classify an agent's action into one of the 9 Pythia outcome levels.

    Args:
        step: Stats for this step (action, cache result, prefetch status).
        bw_high: Whether bandwidth is above the high-BW threshold (70%).

    Returns:
        PrefetchOutcome enum value.
    """
    if not step.prefetch_issued:
        # Agent chose NO_PREFETCH
        if not step.access_predictable:
            return PrefetchOutcome.CORRECT_RESTRAINT   # +8: Smart restraint
        elif bw_high:
            return PrefetchOutcome.NO_PREFETCH_HIGH_BW  # -2: Acceptable conservatism
        else:
            return PrefetchOutcome.NO_PREFETCH_LOW_BW   # -4: Missed opportunity

    # Agent issued a prefetch
    if step.prefetch_was_redundant:
        return PrefetchOutcome.REDUNDANT               # -2: Already in cache

    if step.pollution_this_step:
        return PrefetchOutcome.POLLUTION               # -18: Worst outcome

    if step.prefetch_was_useful:
        if step.prefetch_was_timely:
            return PrefetchOutcome.ACCURATE_TIMELY     # +20: Best outcome
        else:
            return PrefetchOutcome.ACCURATE_LATE       # +12: Right but slow

    # Inaccurate prefetch — penalty depends on bandwidth state
    if bw_high:
        return PrefetchOutcome.INACCURATE_HIGH_BW      # -14: Wrong AND expensive
    else:
        return PrefetchOutcome.INACCURATE_LOW_BW       # -8: Wrong but cheap


def compute_step_reward(
    step: StepStats,
    episode: EpisodeStats,
    bw_high: bool,
) -> Reward:
    """Compute the per-step reward for the agent.

    This function:
      1. Classifies the action into a Pythia outcome
      2. Looks up the raw reward value
      3. Normalizes to [0.001, 0.999]
      4. Breaks down into 6 interpretable components
      5. Generates a human-readable explanation

    Returns:
        Reward object with total, components, outcome, and explanation.
    """
    # Step 1: Classify
    outcome = classify_prefetch_outcome(step, bw_high)
    step.outcome = outcome  # Store for counter tracking

    # Step 2: Raw reward
    raw_reward = REWARD_VALUES[outcome]

    # Step 3: Normalize
    total = normalize_reward(raw_reward)

    # Step 4: Component breakdown
    components = _compute_components(step, episode, outcome)

    # Step 5: Explanation
    explanation = _generate_explanation(step, outcome, raw_reward)

    return Reward(
        total=total,
        components=components,
        outcome=outcome.value,
        explanation=explanation,
    )


def _compute_components(
    step: StepStats,
    episode: EpisodeStats,
    outcome: PrefetchOutcome,
) -> RewardComponents:
    """Break down the reward into 6 named components.

    Components are designed to give the agent fine-grained feedback
    about WHAT aspect of its decision was good or bad.
    """
    comp = RewardComponents()

    # ── ACCURACY (0.0–0.25) ──────────────────────────────────────────
    if outcome == PrefetchOutcome.ACCURATE_TIMELY:
        comp.accuracy = 0.25
    elif outcome == PrefetchOutcome.ACCURATE_LATE:
        comp.accuracy = 0.18
    elif outcome == PrefetchOutcome.REDUNDANT:
        comp.accuracy = 0.05  # at least the right address
    elif outcome in (PrefetchOutcome.INACCURATE_LOW_BW, PrefetchOutcome.INACCURATE_HIGH_BW):
        comp.accuracy = 0.0
    elif outcome == PrefetchOutcome.POLLUTION:
        comp.accuracy = 0.0
    else:
        # NO_PREFETCH actions — use running accuracy
        if episode.total_prefetches_issued > 0:
            running = episode.useful_prefetches / episode.total_prefetches_issued
            comp.accuracy = running * 0.25
        else:
            comp.accuracy = 0.125  # neutral

    # ── COVERAGE (0.0–0.20) ──────────────────────────────────────────
    if step.demand_hit_from_prefetch:
        comp.coverage = 0.20  # demand hit satisfied by prior prefetch
    elif step.hit:
        comp.coverage = 0.10  # natural cache hit
    else:
        comp.coverage = 0.0   # miss

    # ── POLLUTION CONTROL (0.0–0.20) ─────────────────────────────────
    if outcome == PrefetchOutcome.POLLUTION:
        comp.pollution = 0.0  # ZERO for causing pollution
    elif episode.total_prefetches_issued > 0:
        pr = episode.pollution_evictions / episode.total_prefetches_issued
        comp.pollution = max(0.0, (1.0 - pr * 5)) * 0.20
    else:
        comp.pollution = 0.20  # no prefetches = no pollution

    # ── BANDWIDTH (0.0–0.15) ─────────────────────────────────────────
    bw = step.bandwidth_utilization
    if bw < 0.50:
        comp.bandwidth = 0.15
    elif bw < 0.70:
        comp.bandwidth = 0.15 * (1.0 - (bw - 0.50) / 0.40)
    elif bw < 0.90:
        comp.bandwidth = 0.15 * 0.5 * (1.0 - (bw - 0.70) / 0.20)
    else:
        comp.bandwidth = 0.0  # critical

    # ── TIMELINESS (0.0–0.10) ────────────────────────────────────────
    if step.demand_hit_from_prefetch and step.prefetch_was_timely:
        comp.timeliness = 0.10
    elif step.demand_hit_from_prefetch:
        comp.timeliness = 0.03  # late but partially useful
    else:
        comp.timeliness = 0.0

    # ── RESTRAINT (0.0–0.10) ─────────────────────────────────────────
    if outcome == PrefetchOutcome.CORRECT_RESTRAINT:
        comp.restraint = 0.10
    elif outcome in (PrefetchOutcome.NO_PREFETCH_LOW_BW, PrefetchOutcome.NO_PREFETCH_HIGH_BW):
        comp.restraint = 0.02  # at least no harm

    return comp


def _generate_explanation(
    step: StepStats,
    outcome: PrefetchOutcome,
    raw_reward: float,
) -> str:
    """Generate a human-readable explanation of the reward.

    This is included in the observation so the LLM can understand
    WHY its last action succeeded or failed.
    """
    explanations = {
        PrefetchOutcome.ACCURATE_TIMELY: (
            "Excellent! Prefetch was accurate AND arrived before the demand request. "
            f"Raw: +{raw_reward:.0f}"
        ),
        PrefetchOutcome.ACCURATE_LATE: (
            "Prefetch was accurate but arrived LATE — the demand already started. "
            f"Consider prefetching earlier. Raw: +{raw_reward:.0f}"
        ),
        PrefetchOutcome.CORRECT_RESTRAINT: (
            "Smart restraint! Correctly chose NO_PREFETCH on an unpredictable access. "
            f"Raw: +{raw_reward:.0f}"
        ),
        PrefetchOutcome.REDUNDANT: (
            "Wasted effort — that line was already in cache. "
            f"Raw: {raw_reward:.0f}"
        ),
        PrefetchOutcome.NO_PREFETCH_HIGH_BW: (
            "Conservative under pressure — acceptable since bandwidth was high. "
            f"Raw: {raw_reward:.0f}"
        ),
        PrefetchOutcome.NO_PREFETCH_LOW_BW: (
            "Missed opportunity — bandwidth was available but you didn't prefetch. "
            f"Raw: {raw_reward:.0f}"
        ),
        PrefetchOutcome.INACCURATE_LOW_BW: (
            "Wrong prediction. Prefetched data was never used. "
            f"Bandwidth impact: low. Raw: {raw_reward:.0f}"
        ),
        PrefetchOutcome.INACCURATE_HIGH_BW: (
            "Wrong prediction AND bandwidth was already stressed. "
            f"This is costly. Raw: {raw_reward:.0f}"
        ),
        PrefetchOutcome.POLLUTION: (
            "POLLUTION! Your prefetch evicted a useful line from cache. "
            f"This is the worst possible outcome. Raw: {raw_reward:.0f}"
        ),
    }
    return explanations.get(outcome, f"Outcome: {outcome.value}, Raw: {raw_reward:.0f}")
