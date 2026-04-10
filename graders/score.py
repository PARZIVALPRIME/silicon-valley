"""Final episode grader — deterministic scoring for all metrics.

Every formula is pure math.  No LLM grading.  No randomness.
Same EpisodeStats input → same scores output, every time.

Metrics computed:
  accuracy      — useful_prefetches / total_prefetches
  coverage      — (baseline_misses - agent_misses) / baseline_misses
  pollution     — 1.0 - (pollution_evictions / total_prefetches) × 5
  bandwidth     — sweet-spot scoring (0.3-0.7 utilization is optimal)
  timeliness    — timely_prefetches / useful_prefetches
  restraint     — correct_no_prefetch / unpredictable_accesses
  fairness      — Jain's Fairness Index over per-workload hit rates
  sla           — priority-weighted SLA compliance (0, 0.5, or 1.0 per workload)
  adaptation    — accuracy recovery speed after regime change
  cross_core_pollution — cross-core eviction rate (2× penalty)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from engine.counters import EpisodeStats
from tasks.base import BaseTask


def clamp(v: float) -> float:
    """All scores strictly between 0 and 1 for OpenEnv compliance."""
    return round(max(0.01, min(0.99, v)), 3)


def compute_final_score(
    stats: EpisodeStats,
    task: BaseTask,
) -> Dict[str, float]:
    """Compute the final episode score as a weighted sum of metrics.

    Returns dict with each metric name → clamped score, plus 'total'.
    """
    weights = task.grader_weights()
    scores: Dict[str, float] = {}

    # ── ACCURACY ─────────────────────────────────────────────────────
    if "accuracy" in weights:
        if stats.total_prefetches_issued > 0:
            scores["accuracy"] = stats.useful_prefetches / stats.total_prefetches_issued
        else:
            scores["accuracy"] = 0.0

    # ── COVERAGE ─────────────────────────────────────────────────────
    if "coverage" in weights:
        if stats.baseline_misses > 0:
            eliminated = stats.baseline_misses - stats.total_misses
            scores["coverage"] = max(0.0, eliminated / stats.baseline_misses)
        else:
            scores["coverage"] = 1.0  # no misses in baseline = perfect

    # ── POLLUTION ────────────────────────────────────────────────────
    if "pollution" in weights:
        if stats.total_prefetches_issued > 0:
            ratio = stats.pollution_evictions / stats.total_prefetches_issued
            scores["pollution"] = max(0.0, 1.0 - ratio * 5.0)
        else:
            scores["pollution"] = 1.0

    # ── BANDWIDTH ────────────────────────────────────────────────────
    if "bandwidth" in weights:
        avg_bw = stats.avg_bandwidth_utilization
        if avg_bw < 0.30:
            scores["bandwidth"] = avg_bw / 0.30 * 0.80  # slight penalty for underuse
        elif avg_bw < 0.70:
            scores["bandwidth"] = 1.0  # sweet spot
        else:
            scores["bandwidth"] = max(0.0, 1.0 - (avg_bw - 0.70) / 0.30)

    # ── TIMELINESS ───────────────────────────────────────────────────
    if "timeliness" in weights:
        if stats.useful_prefetches > 0:
            scores["timeliness"] = stats.timely_prefetches / stats.useful_prefetches
        else:
            scores["timeliness"] = 0.0

    # ── RESTRAINT ────────────────────────────────────────────────────
    if "restraint" in weights:
        if stats.unpredictable_accesses > 0:
            scores["restraint"] = stats.correct_no_prefetch / stats.unpredictable_accesses
        else:
            scores["restraint"] = 1.0

    # ── FAIRNESS (Jain's Fairness Index) ─────────────────────────────
    if "fairness" in weights:
        hit_rates = list(stats.per_workload_hit_rates.values())
        if len(hit_rates) > 1:
            n = len(hit_rates)
            sum_x = sum(hit_rates)
            sum_x2 = sum(x * x for x in hit_rates)
            if sum_x2 > 0:
                scores["fairness"] = (sum_x ** 2) / (n * sum_x2)
            else:
                scores["fairness"] = 0.0
        else:
            scores["fairness"] = 1.0  # single workload = trivially fair

    # ── SLA COMPLIANCE ───────────────────────────────────────────────
    if "sla" in weights:
        sla_reqs = task.sla_requirements()
        if sla_reqs:
            sla_scores: List[tuple[float, float]] = []  # (score, priority_weight)
            priority_weights = {"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.2}

            for sla in sla_reqs:
                wid = sla.workload_id
                ws = stats.per_workload.get(wid)
                pw = priority_weights.get(sla.priority, 0.33)

                if ws is None:
                    sla_scores.append((0.0, pw))
                    continue

                latency_ok = ws.avg_latency <= sla.max_avg_latency
                hit_ok = ws.hit_rate >= sla.min_hit_rate

                if latency_ok and hit_ok:
                    sla_scores.append((1.0, pw))
                elif (ws.avg_latency <= sla.max_avg_latency * 1.5 and
                      ws.hit_rate >= sla.min_hit_rate * 0.8):
                    sla_scores.append((0.5, pw))  # partial compliance
                else:
                    sla_scores.append((0.0, pw))

            total_weight = sum(pw for _, pw in sla_scores)
            if total_weight > 0:
                scores["sla"] = sum(s * pw for s, pw in sla_scores) / total_weight
            else:
                scores["sla"] = 0.0
        else:
            scores["sla"] = 1.0  # no SLAs

    # ── ADAPTATION (regime change recovery) ──────────────────────────
    if "adaptation" in weights:
        if task.regime_changes():
            before = stats.accuracy_before_regime
            after = stats.accuracy_after_regime_20
            if before > 0.10:
                scores["adaptation"] = min(1.0, after / before)
            else:
                scores["adaptation"] = min(1.0, after / 0.10)
        else:
            scores["adaptation"] = 1.0

    # ── CROSS-CORE POLLUTION ─────────────────────────────────────────
    if "cross_core_pollution" in weights:
        if stats.total_prefetches_issued > 0:
            # 2× penalty multiplier for cross-core pollution
            ratio = (stats.cross_core_pollution * 2) / stats.total_prefetches_issued
            scores["cross_core_pollution"] = max(0.0, 1.0 - ratio * 10.0)
        else:
            scores["cross_core_pollution"] = 1.0

    # ── WEIGHTED TOTAL ───────────────────────────────────────────────
    total = 0.0
    for metric, weight in weights.items():
        if metric in scores:
            total += scores[metric] * weight

    # Clamp all scores
    result = {k: clamp(v) for k, v in scores.items()}
    result["total"] = clamp(total)

    return result
