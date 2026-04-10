"""Performance counters and episode statistics tracking.

Tracks every metric needed by the graders:
  - Prefetch lifecycle (RAT/RAL/RIN_L/RIN_H/POLLUTION/REDUNDANT/RESTRAINT)
  - Cache hit/miss totals per level
  - Bandwidth and MSHR statistics
  - Per-workload breakdowns (for fairness scoring)
  - Regime-change adaptation tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import PrefetchOutcome


# ───────────────────────────────────────────────────────────────────────────
# Per-Step Stats (one memory access)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class StepStats:
    """Statistics for a single step (one memory access)."""

    step: int = 0

    # Access info
    workload_id: str = ""
    phase: str = ""
    address: int = 0
    pc: int = 0
    is_write: bool = False

    # Cache result
    hit: bool = False
    hit_level: str = "MISS"     # L1 | L2 | LLC | MISS
    latency: int = 0

    # Agent's action this step
    action_type: str = ""       # "prefetch" | "no_prefetch" | etc.
    prefetch_offsets: List[int] = field(default_factory=list)

    # Prefetch outcome classification (Pythia 9-level)
    outcome: PrefetchOutcome = PrefetchOutcome.NO_PREFETCH_LOW_BW
    prefetch_issued: bool = False
    prefetch_was_useful: bool = False
    prefetch_was_redundant: bool = False
    prefetch_was_timely: bool = False   # Arrived before demand
    pollution_this_step: bool = False
    cross_core_pollution: bool = False

    # System state at this step
    bandwidth_utilization: float = 0.0
    mshr_utilization: float = 0.0
    mshr_stall: bool = False

    # Pattern info
    access_predictable: bool = True     # False for random/hash phases
    delta: int = 0                      # addr[n] - addr[n-1] in cache lines

    # Was this a demand hit from a PRIOR prefetch?
    demand_hit_from_prefetch: bool = False


# ───────────────────────────────────────────────────────────────────────────
# Per-Workload Stats (for fairness / SLA scoring)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class WorkloadStats:
    """Aggregate statistics for one workload within an episode."""

    workload_id: str = ""
    total_accesses: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_prefetches: int = 0
    useful_prefetches: int = 0
    pollution_evictions: int = 0
    total_latency_cycles: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_hits / self.total_accesses

    @property
    def avg_latency(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_latency_cycles / self.total_accesses


# ───────────────────────────────────────────────────────────────────────────
# Episode Stats (aggregate for final grading)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    """Aggregate statistics for an entire episode.

    This is the input to the final grader (graders/score.py).
    Every field is updated incrementally during the episode.
    """

    # ── Prefetch lifecycle (Pythia 9-level) ──────────────────────────────
    total_prefetches_issued: int = 0
    accurate_timely: int = 0           # RAT
    accurate_late: int = 0             # RAL
    redundant: int = 0                 # Already in cache
    inaccurate_low_bw: int = 0         # RIN-L
    inaccurate_high_bw: int = 0        # RIN-H
    pollution_evictions: int = 0        # Prefetch evicted useful line
    cross_core_pollution: int = 0       # Cross-core LLC eviction
    correct_no_prefetch: int = 0        # Smart NO_PREFETCH on unpredictable
    missed_opportunity: int = 0         # NO_PREFETCH when pattern was predictable
    mshr_drops: int = 0                 # Prefetch dropped due to full MSHR

    @property
    def useful_prefetches(self) -> int:
        return self.accurate_timely + self.accurate_late

    # ── Access statistics ────────────────────────────────────────────────
    total_accesses: int = 0
    total_hits: int = 0
    total_misses: int = 0

    # ── Bandwidth ────────────────────────────────────────────────────────
    bandwidth_samples: int = 0
    bandwidth_sum: float = 0.0
    peak_bandwidth: float = 0.0
    bandwidth_drops: int = 0

    @property
    def avg_bandwidth_utilization(self) -> float:
        if self.bandwidth_samples == 0:
            return 0.0
        return self.bandwidth_sum / self.bandwidth_samples

    # ── MSHR ─────────────────────────────────────────────────────────────
    mshr_stalls: int = 0
    demand_blocked_by_prefetch: int = 0

    # ── Timeliness ───────────────────────────────────────────────────────
    timely_prefetches: int = 0

    # ── Pattern / Restraint ──────────────────────────────────────────────
    unpredictable_accesses: int = 0

    # ── Per-workload breakdown ───────────────────────────────────────────
    per_workload: Dict[str, WorkloadStats] = field(default_factory=dict)

    # ── Regime change tracking ───────────────────────────────────────────
    accuracy_before_regime: float = 0.0
    accuracy_after_regime_20: float = 0.0
    regime_change_step: Optional[int] = None

    # ── Baseline comparison ──────────────────────────────────────────────
    baseline_misses: int = 0   # Misses with zero prefetching (computed in reset)

    # ── Reward tracking ──────────────────────────────────────────────────
    total_reward: float = 0.0
    step_rewards: List[float] = field(default_factory=list)

    def record_step(self, step_stats: StepStats) -> None:
        """Update all counters from a completed step."""
        self.total_accesses += 1

        if step_stats.hit:
            self.total_hits += 1
        else:
            self.total_misses += 1

        # Prefetch lifecycle
        if step_stats.prefetch_issued:
            self.total_prefetches_issued += 1
            outcome = step_stats.outcome

            if outcome == PrefetchOutcome.ACCURATE_TIMELY:
                self.accurate_timely += 1
                self.timely_prefetches += 1
            elif outcome == PrefetchOutcome.ACCURATE_LATE:
                self.accurate_late += 1
            elif outcome == PrefetchOutcome.REDUNDANT:
                self.redundant += 1
            elif outcome == PrefetchOutcome.INACCURATE_LOW_BW:
                self.inaccurate_low_bw += 1
            elif outcome == PrefetchOutcome.INACCURATE_HIGH_BW:
                self.inaccurate_high_bw += 1
            elif outcome == PrefetchOutcome.POLLUTION:
                self.pollution_evictions += 1

        else:
            # NO_PREFETCH action
            if step_stats.outcome == PrefetchOutcome.CORRECT_RESTRAINT:
                self.correct_no_prefetch += 1
            elif step_stats.outcome == PrefetchOutcome.NO_PREFETCH_LOW_BW:
                self.missed_opportunity += 1

        # Cross-core pollution
        if step_stats.cross_core_pollution:
            self.cross_core_pollution += 1

        # Pattern tracking
        if not step_stats.access_predictable:
            self.unpredictable_accesses += 1

        # Bandwidth
        self.bandwidth_samples += 1
        self.bandwidth_sum += step_stats.bandwidth_utilization
        self.peak_bandwidth = max(self.peak_bandwidth, step_stats.bandwidth_utilization)

        # MSHR
        if step_stats.mshr_stall:
            self.mshr_stalls += 1

        # Per-workload
        wid = step_stats.workload_id
        if wid:
            if wid not in self.per_workload:
                self.per_workload[wid] = WorkloadStats(workload_id=wid)
            ws = self.per_workload[wid]
            ws.total_accesses += 1
            if step_stats.hit:
                ws.total_hits += 1
            else:
                ws.total_misses += 1
            ws.total_latency_cycles += step_stats.latency
            if step_stats.prefetch_issued:
                ws.total_prefetches += 1
                if step_stats.prefetch_was_useful:
                    ws.useful_prefetches += 1
            if step_stats.pollution_this_step:
                ws.pollution_evictions += 1

    @property
    def per_workload_hit_rates(self) -> Dict[str, float]:
        return {wid: ws.hit_rate for wid, ws in self.per_workload.items()}

    @property
    def per_workload_avg_latency(self) -> Dict[str, float]:
        return {wid: ws.avg_latency for wid, ws in self.per_workload.items()}

    def prefetch_accuracy(self) -> float:
        """Useful prefetches / total prefetches issued."""
        if self.total_prefetches_issued == 0:
            return 0.0
        return self.useful_prefetches / self.total_prefetches_issued

    def coverage(self) -> float:
        """Fraction of baseline misses eliminated."""
        if self.baseline_misses == 0:
            return 1.0
        eliminated = self.baseline_misses - self.total_misses
        return max(0.0, eliminated / self.baseline_misses)

    def pollution_ratio(self) -> float:
        """Pollution evictions / total prefetches."""
        if self.total_prefetches_issued == 0:
            return 0.0
        return self.pollution_evictions / self.total_prefetches_issued
