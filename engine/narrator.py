"""Narrative generator for LLM-friendly observations.

This is SiliconMind's #1 differentiator.  Instead of making the LLM parse
raw hex addresses and numeric arrays, the narrator produces 2-3 sentence
natural language summaries that an LLM can directly reason about.

Examples:
  "Sequential pattern continues (stride-1, confidence 98%). Your last 10
   prefetches were all accurate and timely. Cache hit rate is 92%."

  "⚠️ PHASE TRANSITION: Your accuracy crashed from 85% to 11%. Deltas
   changed from [4,4,4] to [256,256,256]. This is likely forward→backward
   pass in ML training. Consider stride-256 prefetching or NO_PREFETCH
   until the new pattern stabilizes."
"""

from __future__ import annotations

from typing import Dict, List, Optional


def generate_narrative(
    step: int,
    steps_remaining: int,
    workload: str,
    phase: str,
    cache_result: str,
    detected_pattern: str,
    pattern_confidence: float,
    previous_pattern: str,
    previous_confidence: float,
    prefetch_accuracy_recent: float,
    prefetch_accuracy_total: float,
    hit_rate: float,
    bandwidth_pressure: str,
    bandwidth_utilization: float,
    mshr_risk: str,
    pollution_rate: float,
    last_outcome: str,
    last_explanation: str,
    active_workloads: List[str],
    sla_status: Dict[str, str],
    recent_deltas: List[int],
) -> str:
    """Generate a 2-3 sentence narrative from current state.

    The narrative prioritizes the most actionable information:
      1. Phase transitions and pattern changes (urgent)
      2. System pressure warnings (bandwidth/MSHR/pollution)
      3. Performance summary (accuracy, hit rate)
      4. Multi-tenant SLA warnings
    """
    parts: List[str] = []

    # ── Header: step + workload context ──────────────────────────────────
    header = f"Step {step}"
    if steps_remaining > 0:
        header += f"/{step + steps_remaining}"
    if workload and workload != "default":
        header += f" | {workload}"
    if phase and phase != "unknown":
        header += f" ({phase})"
    header += f" | {cache_result}"
    parts.append(header + ".")

    # ── Priority 1: Phase transition detection ───────────────────────────
    if (
        previous_pattern
        and previous_pattern != detected_pattern
        and previous_confidence > 0.6
        and pattern_confidence < 0.5
    ):
        parts.append(
            f"⚠️ PHASE TRANSITION: Pattern changed from '{previous_pattern}' "
            f"(confidence {previous_confidence:.0%}) to '{detected_pattern}' "
            f"(confidence {pattern_confidence:.0%}). "
            f"Recent deltas: {recent_deltas[-4:]}."
        )
        # Accuracy crash detection
        if prefetch_accuracy_total > 0.5 and prefetch_accuracy_recent < 0.2:
            parts.append(
                f"Your accuracy crashed from {prefetch_accuracy_total:.0%} to "
                f"{prefetch_accuracy_recent:.0%}. Consider NO_PREFETCH until "
                f"the new pattern stabilizes."
            )
    elif pattern_confidence < 0.3 and step > 10:
        parts.append(
            f"Pattern is uncertain ('{detected_pattern}', confidence "
            f"{pattern_confidence:.0%}). Deltas: {recent_deltas[-4:]}. "
            f"Consider NO_PREFETCH or reduced aggressiveness."
        )

    # ── Priority 2: System pressure warnings ─────────────────────────────
    warnings: List[str] = []
    if bandwidth_pressure in ("high", "critical"):
        warnings.append(
            f"Bandwidth {bandwidth_pressure.upper()} ({bandwidth_utilization:.0%})"
        )
    if mshr_risk == "stalling":
        warnings.append("MSHR stalling — prefetches blocking demand requests")
    if pollution_rate > 0.10:
        warnings.append(
            f"Pollution rate {pollution_rate:.0%} — prefetches evicting useful data"
        )

    if warnings:
        parts.append("⚠️ " + ". ".join(warnings) + ".")

    # ── Priority 3: Performance summary ──────────────────────────────────
    if not any("PHASE TRANSITION" in p for p in parts):
        if pattern_confidence > 0.7:
            parts.append(
                f"Pattern: {detected_pattern} (confidence {pattern_confidence:.0%}). "
                f"Accuracy: {prefetch_accuracy_recent:.0%} recent, "
                f"{prefetch_accuracy_total:.0%} total. "
                f"Hit rate: {hit_rate:.0%}."
            )
        elif last_outcome:
            parts.append(
                f"Last action: {last_outcome}. {last_explanation}"
            )

    # ── Priority 4: Multi-tenant SLA warnings ────────────────────────────
    if len(active_workloads) > 1:
        violated = [w for w, s in sla_status.items() if s == "violated"]
        at_risk = [w for w, s in sla_status.items() if s == "at_risk"]
        if violated:
            parts.append(
                f"🚨 SLA VIOLATED for: {', '.join(violated)}. "
                f"Immediate action required."
            )
        elif at_risk:
            parts.append(
                f"⚠️ SLA at risk for: {', '.join(at_risk)}. "
                f"Consider cache partitioning or bandwidth throttling."
            )

    return " ".join(parts)


def generate_action_hint(
    bandwidth_pressure: str,
    mshr_risk: str,
    pollution_rate: float,
    prefetch_accuracy_recent: float,
    pattern_confidence: float,
    active_workloads: List[str],
    sla_status: Dict[str, str],
) -> str:
    """Generate a short, actionable hint for the agent.

    This is a softer signal than the narrative — just a nudge.
    """
    hints: List[str] = []

    if pollution_rate > 0.15:
        hints.append("Reduce aggressiveness — high pollution rate")
    if bandwidth_pressure == "critical":
        hints.append("Throttle bandwidth — bus is saturated")
    if mshr_risk == "stalling":
        hints.append("Reduce prefetch volume — MSHR stalling")
    if prefetch_accuracy_recent < 0.15 and pattern_confidence < 0.3:
        hints.append("Consider NO_PREFETCH — pattern is unpredictable")
    if prefetch_accuracy_recent > 0.85 and pattern_confidence > 0.8:
        hints.append("Pattern is strong — consider increasing aggressiveness")

    # Multi-tenant hints
    violated = [w for w, s in sla_status.items() if s == "violated"]
    if violated:
        hints.append(
            f"Partition LLC to protect {violated[0]} — SLA violated"
        )

    if hints:
        return hints[0]  # Return most important hint
    return ""
