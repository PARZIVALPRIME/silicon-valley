"""SiliconMind Environment — OpenEnv-compliant memory subsystem simulator.

This is the central orchestrator.  It implements the three required methods:
  reset(task_id, seed)  → initial observation
  step(action)          → observation with reward
  state                 → current internal state

Flow per step:
  1. Read next MemoryAccess from pre-generated trace
  2. Process agent's action (prefetch/no_prefetch/config changes)
  3. Simulate cache hierarchy (demand access → hit/miss)
  4. Simulate bandwidth and MSHR effects
  5. Classify action outcome (Pythia 9-level)
  6. Compute per-step reward
  7. Track episode statistics
  8. Build observation with narrative
  9. If done → compute final grader score
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from engine.bandwidth import BandwidthConfig, BandwidthController
from engine.cache import CacheConfig, CacheHierarchy, LINE_SIZE
from engine.counters import EpisodeStats, StepStats
from engine.mshr import MSHRConfig, MSHRQueue
from engine.narrator import generate_action_hint, generate_narrative
from graders.baselines import compute_baseline_misses
from graders.reward import classify_prefetch_outcome, compute_step_reward
from graders.score import compute_final_score
from models import (
    ActionType,
    BandwidthPressure,
    BandwidthState,
    CacheLevel,
    CacheStats,
    MSHRRisk,
    MSHRState,
    PatternAnalysis,
    Reward,
    SLAStatus,
    SiliconMindAction,
    SiliconMindObservation,
    SiliconMindState,
)
from tasks import DEFAULT_TASK, TASK_REGISTRY
from tasks.base import BaseTask
from workloads.base import MemoryAccess


# ────────────────────────────────────────────────────────────────────────────
# Pattern Detector — identifies stride patterns from recent deltas
# ────────────────────────────────────────────────────────────────────────────

class PatternDetector:
    """Lightweight online pattern detector for the observation.

    Tracks the last N deltas and identifies dominant stride patterns.
    Outputs: detected_pattern, confidence, pattern_age.
    """

    def __init__(self, window_size: int = 8) -> None:
        self.window_size = window_size
        self._deltas: List[int] = []
        self._pattern: str = "unknown"
        self._confidence: float = 0.0
        self._pattern_age: int = 0
        self._previous_pattern: str = ""
        self._previous_confidence: float = 0.0

    def update(self, delta: int) -> None:
        """Add a new delta and re-detect pattern."""
        self._deltas.append(delta)
        if len(self._deltas) > self.window_size:
            self._deltas.pop(0)

        if len(self._deltas) < 3:
            self._pattern = "unknown"
            self._confidence = 0.0
            return

        # Count how many deltas match the most common value
        from collections import Counter
        counter = Counter(self._deltas)
        most_common_delta, most_common_count = counter.most_common(1)[0]

        new_confidence = most_common_count / len(self._deltas)
        new_pattern = f"stride-{most_common_delta}" if most_common_delta != 0 else "random"

        if new_confidence < 0.3:
            new_pattern = "random"

        # Detect pattern change
        if new_pattern != self._pattern:
            self._previous_pattern = self._pattern
            self._previous_confidence = self._confidence
            self._pattern = new_pattern
            self._pattern_age = 0
        else:
            self._pattern_age += 1

        self._confidence = round(new_confidence, 2)

    @property
    def analysis(self) -> PatternAnalysis:
        return PatternAnalysis(
            detected_pattern=self._pattern,
            confidence=self._confidence,
            pattern_age=self._pattern_age,
            previous_pattern=self._previous_pattern,
            previous_confidence=self._previous_confidence,
        )

    def reset(self) -> None:
        self._deltas.clear()
        self._pattern = "unknown"
        self._confidence = 0.0
        self._pattern_age = 0
        self._previous_pattern = ""
        self._previous_confidence = 0.0


# ────────────────────────────────────────────────────────────────────────────
# Prefetch Tracker — tracks issued prefetches for usefulness evaluation
# ────────────────────────────────────────────────────────────────────────────

class PrefetchTracker:
    """Tracks outstanding prefetches to classify them as useful/wasted.

    When a prefetch is issued, we record the address.
    When a demand access hits a prefetched address, we mark it as useful.
    On expiry (200 cycles), unmarked prefetches are classified as wasted.
    """

    def __init__(self, expiry: int = 200) -> None:
        self.expiry = expiry
        self._pending: Dict[int, Dict] = {}  # addr → {cycle, issued_cycle}
        self.total_issued: int = 0
        self.total_useful: int = 0
        self.total_timely: int = 0
        self.total_wasted: int = 0
        self._recent_outcomes: List[bool] = []  # last 10: True=useful

    def record_issue(self, addr: int, cycle: int) -> None:
        aligned = (addr // LINE_SIZE) * LINE_SIZE
        self._pending[aligned] = {"cycle": cycle}
        self.total_issued += 1

    def check_demand(self, addr: int, cycle: int) -> tuple[bool, bool]:
        """Check if a demand access matches a pending prefetch.

        Returns (was_prefetched: bool, was_timely: bool).
        """
        aligned = (addr // LINE_SIZE) * LINE_SIZE
        if aligned in self._pending:
            entry = self._pending.pop(aligned)
            self.total_useful += 1
            # Timely = data was already in cache before demand
            timely = True  # In our sim, if it's in _pending it was filled
            if timely:
                self.total_timely += 1
            self._recent_outcomes.append(True)
            if len(self._recent_outcomes) > 10:
                self._recent_outcomes.pop(0)
            return True, timely
        return False, False

    def expire(self, cycle: int) -> int:
        """Expire old prefetches. Returns count of expired (wasted)."""
        expired = []
        for addr, info in self._pending.items():
            if cycle - info["cycle"] > self.expiry:
                expired.append(addr)
        for addr in expired:
            del self._pending[addr]
            self.total_wasted += 1
            self._recent_outcomes.append(False)
            if len(self._recent_outcomes) > 10:
                self._recent_outcomes.pop(0)
        return len(expired)

    @property
    def recent_accuracy(self) -> float:
        if not self._recent_outcomes:
            return 0.0
        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    @property
    def total_accuracy(self) -> float:
        if self.total_issued == 0:
            return 0.0
        return self.total_useful / self.total_issued

    def reset(self) -> None:
        self._pending.clear()
        self.total_issued = 0
        self.total_useful = 0
        self.total_timely = 0
        self.total_wasted = 0
        self._recent_outcomes.clear()


# ────────────────────────────────────────────────────────────────────────────
# Main Environment
# ────────────────────────────────────────────────────────────────────────────

class SiliconMindEnvironment(Environment):
    """OpenEnv-compatible memory subsystem intelligence environment."""

    def __init__(self) -> None:
        super().__init__()
        # Task & trace state
        self._task: Optional[BaseTask] = None
        self._task_id: str = ""
        self._seed: int = 0
        self._trace: List[MemoryAccess] = []
        self._trace_idx: int = 0
        self._done: bool = True
        self._cycle: int = 0

        # Hardware simulation
        self._cache: Optional[CacheHierarchy] = None
        self._bandwidth: Optional[BandwidthController] = None
        self._mshr: Optional[MSHRQueue] = None

        # Tracking
        self._episode_stats = EpisodeStats()
        self._pattern_detector = PatternDetector()
        self._prefetch_tracker = PrefetchTracker()
        self._prev_address: int = 0

        # Agent config
        self._aggressiveness: int = 2
        self._bw_throttle: float = 1.0
        self._llc_partition: Dict[str, int] = {}

        # History for observation
        self._recent_deltas: List[int] = []
        self._recent_pcs: List[str] = []
        self._recent_cache_results: List[str] = []
        self._last_outcome: str = ""
        self._last_explanation: str = ""

    @property
    def available_tasks(self) -> List[str]:
        return list(TASK_REGISTRY.keys())

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: str = DEFAULT_TASK,
        **kwargs: Any,
    ) -> SiliconMindObservation:
        """Initialize a new episode with the given task and seed."""
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")

        self._task = TASK_REGISTRY[task_id]
        self._task_id = task_id
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._done = False
        self._cycle = 0
        self._trace_idx = 0

        # Generate trace
        self._trace = self._task.build_trace(self._seed)

        # Initialize hardware
        configs = self._task.cache_configs()
        l1 = configs[0] if len(configs) > 0 else None
        l2 = configs[1] if len(configs) > 1 else None
        llc = configs[2] if len(configs) > 2 else None
        self._cache = CacheHierarchy(l1_config=l1, l2_config=l2, llc_config=llc)

        bw_config = self._task.bandwidth_config()
        self._bandwidth = BandwidthController(bw_config)

        mshr_config = self._task.mshr_config()
        self._mshr = MSHRQueue(mshr_config)

        # Reset trackers
        self._episode_stats = EpisodeStats()
        self._pattern_detector = PatternDetector()
        self._prefetch_tracker = PrefetchTracker()
        self._prev_address = 0
        self._aggressiveness = 2
        self._bw_throttle = 1.0
        self._llc_partition = {}
        self._recent_deltas = []
        self._recent_pcs = []
        self._recent_cache_results = []
        self._last_outcome = ""
        self._last_explanation = ""

        # Compute baseline (no-prefetch) misses
        self._episode_stats.baseline_misses = compute_baseline_misses(
            self._trace, configs,
        )

        # Build initial observation (before first step)
        return self._build_observation(
            access=self._trace[0] if self._trace else None,
            reward=Reward(),
        )

    def step(self, action: SiliconMindAction, **kwargs: Any) -> SiliconMindObservation:
        """Process one step: agent action + next memory access."""
        if self._done:
            return self._build_observation(access=None, reward=Reward())

        if self._trace_idx >= len(self._trace):
            self._done = True
            return self._build_observation(access=None, reward=Reward())

        # ── 1. Read current memory access ────────────────────────────
        access = self._trace[self._trace_idx]
        self._trace_idx += 1
        self._cycle += 10  # approximate cycle increment

        # ── 2. Process agent action ──────────────────────────────────
        step_stats = StepStats(
            step=self._trace_idx,
            workload_id=access.workload_id,
            phase=access.phase,
            address=access.address,
            pc=access.pc,
            is_write=access.is_write,
            access_predictable=access.predictable,
        )

        prefetch_results = self._process_action(action, access, step_stats)

        # ── 3. Simulate demand cache access ──────────────────────────
        # Check if this access was satisfied by a prior prefetch
        was_prefetched, was_timely = self._prefetch_tracker.check_demand(
            access.address, self._cycle,
        )
        step_stats.demand_hit_from_prefetch = was_prefetched
        step_stats.prefetch_was_timely = was_timely

        cache_result = self._cache.access(
            addr=access.address,
            cycle=self._cycle,
            workload_id=access.workload_id,
        )
        step_stats.hit = cache_result.hit
        step_stats.hit_level = cache_result.hit_level
        step_stats.latency = cache_result.total_latency
        step_stats.pollution_this_step = cache_result.pollution_detected

        # ── 4. Bandwidth simulation ──────────────────────────────────
        if not cache_result.hit:
            # Demand miss → issue memory request
            self._bandwidth.request(is_prefetch=False, workload_id=access.workload_id)
        self._bandwidth.tick(cycles=10)
        step_stats.bandwidth_utilization = self._bandwidth.utilization

        # ── 5. Expire old prefetches ─────────────────────────────────
        self._prefetch_tracker.expire(self._cycle)

        # ── 6. MSHR simulation ───────────────────────────────────────
        completed = self._mshr.tick(self._cycle)
        step_stats.mshr_utilization = self._mshr.utilization
        step_stats.mshr_stall = self._mshr.stall_risk == MSHRRisk.STALLING

        # ── 7. Pattern detection ─────────────────────────────────────
        delta = (access.address - self._prev_address) // LINE_SIZE if self._prev_address else 0
        step_stats.delta = delta
        self._pattern_detector.update(delta)
        self._prev_address = access.address

        # Update history
        self._recent_deltas.append(delta)
        if len(self._recent_deltas) > 8:
            self._recent_deltas.pop(0)
        self._recent_pcs.append(f"0x{access.pc:06X}")
        if len(self._recent_pcs) > 8:
            self._recent_pcs.pop(0)
        cache_result_str = cache_result.hit_level
        self._recent_cache_results.append(cache_result_str)
        if len(self._recent_cache_results) > 8:
            self._recent_cache_results.pop(0)

        # ── 8. Compute reward ────────────────────────────────────────
        bw_high = self._bandwidth.is_high_bandwidth
        reward = compute_step_reward(step_stats, self._episode_stats, bw_high)

        # ── 9. Record step stats ─────────────────────────────────────
        self._episode_stats.record_step(step_stats)
        self._episode_stats.total_reward += reward.total
        self._episode_stats.step_rewards.append(reward.total)

        self._last_outcome = reward.outcome
        self._last_explanation = reward.explanation

        # ── 10. Check episode end ────────────────────────────────────
        if self._trace_idx >= len(self._trace):
            self._done = True

        # ── 11. Build observation ────────────────────────────────────
        obs = self._build_observation(access=access, reward=reward)

        # If done, include final scores in info
        if self._done:
            final_scores = compute_final_score(self._episode_stats, self._task)
            obs.info["final_scores"] = final_scores
            obs.info["episode_stats"] = {
                "total_accesses": self._episode_stats.total_accesses,
                "total_hits": self._episode_stats.total_hits,
                "total_misses": self._episode_stats.total_misses,
                "total_prefetches": self._episode_stats.total_prefetches_issued,
                "useful_prefetches": self._episode_stats.useful_prefetches,
                "pollution_evictions": self._episode_stats.pollution_evictions,
                "baseline_misses": self._episode_stats.baseline_misses,
                "prefetch_accuracy": round(self._episode_stats.prefetch_accuracy(), 3),
                "coverage": round(self._episode_stats.coverage(), 3),
            }

        return obs

    @property
    def state(self) -> SiliconMindState:
        """Return current internal state for the /state endpoint."""
        return SiliconMindState(
            task_id=self._task_id,
            step_number=self._trace_idx,
            max_steps=len(self._trace),
            done=self._done,
            total_reward=round(self._episode_stats.total_reward, 3),
            seed=self._seed,
            cache_stats=self._build_cache_stats(),
            bandwidth=self._build_bw_state(),
            mshr=self._build_mshr_state(),
            prefetch_aggressiveness=self._aggressiveness,
            bandwidth_throttle=self._bw_throttle,
            llc_partition=self._llc_partition,
            current_workload=self._trace[self._trace_idx].workload_id if self._trace_idx < len(self._trace) else "",
            current_phase=self._trace[self._trace_idx].phase if self._trace_idx < len(self._trace) else "",
        )

    # ── Action Processing ────────────────────────────────────────────────

    def _process_action(
        self,
        action: SiliconMindAction,
        access: MemoryAccess,
        step_stats: StepStats,
    ) -> None:
        """Process the agent's action and update step_stats accordingly."""
        step_stats.action_type = action.type.value

        if action.type == ActionType.PREFETCH:
            self._process_prefetch(action, access, step_stats)

        elif action.type == ActionType.NO_PREFETCH:
            step_stats.prefetch_issued = False

        elif action.type == ActionType.SET_AGGRESSIVENESS:
            try:
                level = int(action.target)
                self._aggressiveness = max(1, min(4, level))
            except (ValueError, TypeError):
                pass
            step_stats.prefetch_issued = False

        elif action.type == ActionType.THROTTLE_BANDWIDTH:
            try:
                throttle = float(action.target)
                self._bw_throttle = max(0.0, min(1.0, throttle))
            except (ValueError, TypeError):
                pass
            step_stats.prefetch_issued = False

        elif action.type == ActionType.PARTITION_CACHE:
            self._process_partition(action)
            step_stats.prefetch_issued = False

    def _process_prefetch(
        self,
        action: SiliconMindAction,
        access: MemoryAccess,
        step_stats: StepStats,
    ) -> None:
        """Issue prefetch requests for the given offsets."""
        step_stats.prefetch_issued = True

        # Parse offsets from target (e.g., "1,2,4")
        try:
            offsets = [int(x.strip()) for x in action.target.split(",") if x.strip()]
        except (ValueError, TypeError):
            offsets = [1]  # default: prefetch next line

        # Limit by aggressiveness
        offsets = offsets[:self._aggressiveness]
        step_stats.prefetch_offsets = offsets

        any_useful = False
        any_redundant = False
        any_pollution = False

        for offset in offsets:
            prefetch_addr = access.address + offset * LINE_SIZE

            # Check page boundary (4KB = 64 cache lines)
            page_start = (access.address // 4096) * 4096
            page_end = page_start + 4096
            if prefetch_addr < page_start or prefetch_addr >= page_end:
                continue  # Don't cross page boundary

            # Check bandwidth throttle
            if self._bandwidth.utilization >= self._bw_throttle:
                continue

            # Try MSHR allocation
            mshr_ok = self._mshr.allocate(
                address=prefetch_addr,
                is_prefetch=True,
                cycle=self._cycle,
                workload_id=access.workload_id,
            )
            if not mshr_ok:
                self._episode_stats.mshr_drops += 1
                continue

            # Issue bandwidth request
            accepted, extra_lat = self._bandwidth.request(
                is_prefetch=True,
                workload_id=access.workload_id,
            )
            if not accepted:
                continue

            # Insert into cache
            result = self._cache.prefetch(
                addr=prefetch_addr,
                cycle=self._cycle,
                workload_id=access.workload_id,
            )

            if result.redundant:
                any_redundant = True
            elif result.pollution_risk:
                any_pollution = True

            # Track for later usefulness evaluation
            if result.issued:
                self._prefetch_tracker.record_issue(prefetch_addr, self._cycle)

        # Set step stats based on outcomes
        step_stats.prefetch_was_redundant = any_redundant and not any_useful
        step_stats.pollution_this_step = any_pollution

        # Usefulness will be determined retroactively when demand accesses arrive
        # For now, check if the CURRENT demand was from a prior prefetch
        step_stats.prefetch_was_useful = step_stats.demand_hit_from_prefetch

    def _process_partition(self, action: SiliconMindAction) -> None:
        """Process LLC way partitioning request."""
        try:
            parts = action.target.split(",")
            partition = {}
            for part in parts:
                wid, ways = part.strip().split(":")
                partition[wid.strip()] = int(ways.strip())
            self._llc_partition = partition
        except (ValueError, TypeError):
            pass  # Invalid format, ignore

    # ── Observation Building ─────────────────────────────────────────────

    def _build_observation(
        self,
        access: Optional[MemoryAccess],
        reward: Reward,
    ) -> SiliconMindObservation:
        """Construct the full observation for the agent."""
        pattern = self._pattern_detector.analysis

        # Build SLA status
        sla_status: Dict[str, str] = {}
        if self._task and self._task.sla_requirements():
            for sla in self._task.sla_requirements():
                ws = self._episode_stats.per_workload.get(sla.workload_id)
                if ws is None:
                    sla_status[sla.workload_id] = "met"
                elif ws.avg_latency > sla.max_avg_latency or ws.hit_rate < sla.min_hit_rate:
                    sla_status[sla.workload_id] = "violated"
                elif (ws.avg_latency > sla.max_avg_latency * 0.8 or
                      ws.hit_rate < sla.min_hit_rate * 1.2):
                    sla_status[sla.workload_id] = "at_risk"
                else:
                    sla_status[sla.workload_id] = "met"

        # Generate narrative
        bw_state = self._build_bw_state()
        mshr_state = self._build_mshr_state()
        narrative = generate_narrative(
            step=self._trace_idx,
            steps_remaining=len(self._trace) - self._trace_idx,
            workload=access.workload_id if access else "",
            phase=access.phase if access else "",
            cache_result=self._recent_cache_results[-1] if self._recent_cache_results else "MISS",
            detected_pattern=pattern.detected_pattern,
            pattern_confidence=pattern.confidence,
            previous_pattern=pattern.previous_pattern,
            previous_confidence=pattern.previous_confidence,
            prefetch_accuracy_recent=self._prefetch_tracker.recent_accuracy,
            prefetch_accuracy_total=self._prefetch_tracker.total_accuracy,
            hit_rate=self._cache.hit_rate if self._cache else 0.0,
            bandwidth_pressure=bw_state.pressure.value,
            bandwidth_utilization=bw_state.utilization,
            mshr_risk=mshr_state.stall_risk.value,
            pollution_rate=self._episode_stats.pollution_ratio(),
            last_outcome=self._last_outcome,
            last_explanation=self._last_explanation,
            active_workloads=self._task.workload_ids() if self._task else [],
            sla_status=sla_status,
            recent_deltas=self._recent_deltas,
        )

        action_hint = generate_action_hint(
            bandwidth_pressure=bw_state.pressure.value,
            mshr_risk=mshr_state.stall_risk.value,
            pollution_rate=self._episode_stats.pollution_ratio(),
            prefetch_accuracy_recent=self._prefetch_tracker.recent_accuracy,
            pattern_confidence=pattern.confidence,
            active_workloads=self._task.workload_ids() if self._task else [],
            sla_status=sla_status,
        )

        return SiliconMindObservation(
            narrative=narrative,
            step_number=self._trace_idx,
            steps_remaining=len(self._trace) - self._trace_idx,
            current_pc=f"0x{access.pc:06X}" if access else "0x0",
            current_address=f"0x{access.address:08X}" if access else "0x0",
            current_cache_result=CacheLevel(self._recent_cache_results[-1]) if self._recent_cache_results else CacheLevel.MISS,
            is_write=access.is_write if access else False,
            current_workload=access.workload_id if access else "",
            current_phase=access.phase if access else "",
            recent_deltas=list(self._recent_deltas),
            recent_pcs=list(self._recent_pcs),
            recent_cache_results=list(self._recent_cache_results),
            pattern_analysis=pattern,
            last_action_outcome=self._last_outcome,
            last_action_explanation=self._last_explanation,
            prefetch_accuracy_recent=round(self._prefetch_tracker.recent_accuracy, 3),
            prefetch_accuracy_total=round(self._prefetch_tracker.total_accuracy, 3),
            cache_stats=self._build_cache_stats(),
            bandwidth=bw_state,
            mshr=mshr_state,
            pollution_rate=round(self._episode_stats.pollution_ratio(), 3),
            active_workloads=self._task.workload_ids() if self._task else [],
            workload_sla_status=sla_status,
            llc_partition=self._llc_partition,
            available_actions=[a.value for a in self._task.available_actions()] if self._task else [],
            action_hint=action_hint,
            task_id=self._task_id,
            task_description=self._task.description if self._task else "",
            done=self._done,
            reward=reward.total,
            reward_detail=reward,
            info={},
        )

    def _build_cache_stats(self) -> CacheStats:
        if not self._cache:
            return CacheStats()
        return CacheStats(
            hit_rate=round(self._cache.hit_rate, 3),
            l1_hit_rate=round(self._cache.l1_hit_rate, 3),
            l2_hit_rate=round(self._cache.l2_hit_rate, 3),
            llc_hit_rate=round(self._cache.llc_hit_rate, 3),
            total_hits=self._cache.l1_hits + self._cache.l2_hits + self._cache.llc_hits,
            total_misses=self._cache.l1_misses,
            total_prefetches_issued=self._prefetch_tracker.total_issued,
            total_prefetches_useful=self._prefetch_tracker.total_useful,
            total_prefetches_wasted=self._prefetch_tracker.total_wasted,
            total_pollution_evictions=self._cache.pollution_events,
        )

    def _build_bw_state(self) -> BandwidthState:
        if not self._bandwidth:
            return BandwidthState()
        return BandwidthState(
            utilization=round(self._bandwidth.utilization, 3),
            pressure=self._bandwidth.pressure,
            outstanding_requests=self._bandwidth._outstanding,
            dropped_requests=self._bandwidth.total_drops,
        )

    def _build_mshr_state(self) -> MSHRState:
        if not self._mshr:
            return MSHRState()
        return MSHRState(
            utilization=round(self._mshr.utilization, 3),
            stall_risk=self._mshr.stall_risk,
            total_stalls=self._mshr.total_stalls,
            demand_blocked=self._mshr.demand_blocked,
        )
