"""SiliconMind — Pydantic Models for the OpenEnv Memory Subsystem Intelligence Gym.

All models inherit from openenv.core.env_server base classes to ensure
full OpenEnv spec compliance.  Every field is typed and documented so the
LLM agent (and human reviewers) can reason about the observation space.

Architecture note:
    Action  → what the agent sends
    Observation → what the agent receives (includes reward)
    State → internal episode state (for /state endpoint)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# OpenEnv base classes — same import path used by CasualOps (proven pattern)
# ---------------------------------------------------------------------------
from openenv.core.env_server import (
    Action as OEAction,
    Observation as OEObservation,
    State as OEState,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENUMS                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ActionType(str, Enum):
    """The five control levers available to the agent.

    Easy tasks expose only PREFETCH / NO_PREFETCH.
    Medium adds SET_AGGRESSIVENESS.
    Hard/Extreme unlock THROTTLE_BANDWIDTH and PARTITION_CACHE.
    """

    PREFETCH = "prefetch"
    NO_PREFETCH = "no_prefetch"
    SET_AGGRESSIVENESS = "set_aggressiveness"
    THROTTLE_BANDWIDTH = "throttle_bandwidth"
    PARTITION_CACHE = "partition_cache"


class PrefetchOutcome(str, Enum):
    """Pythia-inspired 9-level classification for every prefetch decision.

    Reference: Pythia (MICRO '21) — Bera et al.
    Extensions: POLLUTION and CORRECT_RESTRAINT are SiliconMind originals.
    """

    # Positive outcomes
    ACCURATE_TIMELY = "RAT"          # Prefetch was used AND arrived before demand
    ACCURATE_LATE = "RAL"            # Prefetch was used BUT arrived after demand started
    CORRECT_RESTRAINT = "RESTRAINT"  # Agent correctly chose NO_PREFETCH on unpredictable

    # Neutral / mild negative
    REDUNDANT = "REDUNDANT"          # Prefetched a line already in cache
    NO_PREFETCH_LOW_BW = "RNP_L"    # Did not prefetch, bandwidth was available (missed opportunity)
    NO_PREFETCH_HIGH_BW = "RNP_H"   # Did not prefetch, bandwidth was congested (acceptable)

    # Negative outcomes
    INACCURATE_LOW_BW = "RIN_L"     # Wrong prefetch, bandwidth was available
    INACCURATE_HIGH_BW = "RIN_H"    # Wrong prefetch, bandwidth was congested (worse)
    POLLUTION = "POLLUTION"          # Prefetch evicted a useful line from cache


class CacheLevel(str, Enum):
    """Where a demand access resolved in the cache hierarchy."""

    L1_HIT = "L1_HIT"
    L2_HIT = "L2_HIT"
    LLC_HIT = "LLC_HIT"
    MISS = "MISS"


class BandwidthPressure(str, Enum):
    """Qualitative bandwidth state for LLM-friendly observation."""

    LOW = "low"           # utilization < 50%
    MODERATE = "moderate"  # 50% - 70%
    HIGH = "high"          # 70% - 90%
    CRITICAL = "critical"  # > 90%


class MSHRRisk(str, Enum):
    """MSHR stall risk level."""

    NONE = "none"              # < 50% utilization
    APPROACHING = "approaching"  # 50% - 80%
    STALLING = "stalling"      # > 80% or actual stall


class SLAStatus(str, Enum):
    """Per-workload SLA compliance status."""

    MET = "met"
    AT_RISK = "at_risk"
    VIOLATED = "violated"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REWARD MODELS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Pythia-inspired reward values (raw, before normalization)
REWARD_VALUES: Dict[PrefetchOutcome, float] = {
    PrefetchOutcome.ACCURATE_TIMELY:   +20.0,
    PrefetchOutcome.ACCURATE_LATE:     +12.0,
    PrefetchOutcome.CORRECT_RESTRAINT: +8.0,
    PrefetchOutcome.REDUNDANT:         -2.0,
    PrefetchOutcome.NO_PREFETCH_LOW_BW:  -4.0,
    PrefetchOutcome.NO_PREFETCH_HIGH_BW: -2.0,
    PrefetchOutcome.INACCURATE_LOW_BW:  -8.0,
    PrefetchOutcome.INACCURATE_HIGH_BW: -14.0,
    PrefetchOutcome.POLLUTION:         -18.0,
}

# For normalization to [0, 1]
_REWARD_MIN = min(REWARD_VALUES.values())  # -18
_REWARD_MAX = max(REWARD_VALUES.values())  # +20


def normalize_reward(raw: float) -> float:
    """Map raw Pythia-scale reward to [0.001, 0.999] for OpenEnv compliance."""
    if _REWARD_MAX == _REWARD_MIN:
        return 0.5
    normalized = (raw - _REWARD_MIN) / (_REWARD_MAX - _REWARD_MIN)
    return round(max(0.001, min(0.999, normalized)), 3)


class RewardComponents(BaseModel):
    """Decomposed per-step reward with 6 named components.

    All components are in [0.0, 0.25] so they sum to at most 1.0.
    The component weights change per-task (defined in task config).
    """

    accuracy: float = Field(default=0.0, ge=0.0, le=0.25,
                            description="Prefetch prediction accuracy")
    coverage: float = Field(default=0.0, ge=0.0, le=0.20,
                            description="Miss elimination rate")
    pollution: float = Field(default=0.0, ge=0.0, le=0.20,
                             description="Cache pollution control")
    bandwidth: float = Field(default=0.0, ge=0.0, le=0.15,
                             description="Memory bus efficiency")
    timeliness: float = Field(default=0.0, ge=0.0, le=0.10,
                              description="Prefetch arrived before demand")
    restraint: float = Field(default=0.0, ge=0.0, le=0.10,
                             description="Correct NO_PREFETCH decisions")

    @property
    def total(self) -> float:
        """Sum of all components, clamped to [0, 1]."""
        return min(1.0, (
            self.accuracy + self.coverage + self.pollution
            + self.bandwidth + self.timeliness + self.restraint
        ))


class Reward(BaseModel):
    """Per-step reward returned to the agent.

    Contains the total normalized scalar, the component breakdown,
    the Pythia classification, and a human-readable explanation.
    """

    total: float = Field(default=0.0, ge=0.0, le=1.0,
                         description="Normalized scalar reward [0, 1]")
    components: RewardComponents = Field(default_factory=RewardComponents)
    outcome: str = Field(default="",
                         description="Pythia classification (RAT/RAL/RIN_L/...)")
    explanation: str = Field(default="",
                             description="Human-readable reward explanation")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVATION SUB-MODELS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PatternAnalysis(BaseModel):
    """Environment-computed pattern detection shared with the agent.

    This is the key LLM-friendly feature: instead of making the LLM
    figure out patterns from raw hex, we tell it what we detected.
    """

    detected_pattern: str = Field(
        default="unknown",
        description="Identified pattern: 'stride-1', 'stride-4', 'random', 'unknown'")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How confident the detector is (drops on phase change)")
    pattern_age: int = Field(
        default=0, ge=0,
        description="Steps since this pattern was first detected")
    previous_pattern: str = Field(
        default="",
        description="What pattern was active before current one")
    previous_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence of the previous pattern")


class CacheStats(BaseModel):
    """Aggregate cache performance counters exposed to the agent."""

    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    l1_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    l2_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    llc_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    total_hits: int = Field(default=0, ge=0)
    total_misses: int = Field(default=0, ge=0)
    total_prefetches_issued: int = Field(default=0, ge=0)
    total_prefetches_useful: int = Field(default=0, ge=0)
    total_prefetches_wasted: int = Field(default=0, ge=0)
    total_pollution_evictions: int = Field(default=0, ge=0)


class BandwidthState(BaseModel):
    """Memory bus state for bandwidth-aware decision making."""

    utilization: float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Current bandwidth utilization [0, 1]")
    pressure: BandwidthPressure = Field(
        default=BandwidthPressure.LOW,
        description="Qualitative pressure level")
    outstanding_requests: int = Field(default=0, ge=0)
    dropped_requests: int = Field(default=0, ge=0,
                                  description="Prefetches dropped due to congestion")


class MSHRState(BaseModel):
    """Miss Status Holding Register state."""

    utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    stall_risk: MSHRRisk = Field(default=MSHRRisk.NONE)
    total_stalls: int = Field(default=0, ge=0,
                              description="Pipeline stalls caused by full MSHR")
    demand_blocked: int = Field(default=0, ge=0,
                                description="Demand requests blocked by prefetch MSHR usage")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ACTION MODEL                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SiliconMindAction(OEAction):
    """Agent → Environment: the prefetch decision.

    Target format varies by action type:
        PREFETCH           → "1,2,4"  (comma-sep cache-line offsets)
        NO_PREFETCH        → "skip"
        SET_AGGRESSIVENESS → "1" | "2" | "3" | "4"
        THROTTLE_BANDWIDTH → "0.6"  (max utilization target)
        PARTITION_CACHE    → "web:6,db:6,video:4"  (LLC way allocation)
    """

    type: ActionType = Field(
        ...,
        description="Which control lever to use")
    target: str = Field(
        ...,
        description="Action parameter (format depends on type)")
    detail: str = Field(
        default="",
        description="Agent's reasoning (optional, for interpretability)")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  OBSERVATION MODEL                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SiliconMindObservation(OEObservation):
    """Environment → Agent: everything the LLM sees each step.

    The `narrative` field is the #1 innovation: a 2-3 sentence natural
    language summary designed for LLM reasoning.  All other fields provide
    structured data for precision.
    """

    # ── LLM-OPTIMIZED NARRATIVE ──────────────────────────────────────────
    narrative: str = Field(
        default="",
        description="2-3 sentence natural language summary of current state")

    # ── CURRENT MEMORY ACCESS ────────────────────────────────────────────
    step_number: int = Field(default=0, ge=0)
    steps_remaining: int = Field(default=0, ge=0)
    current_pc: str = Field(
        default="0x0",
        description="Program counter (hex string)")
    current_address: str = Field(
        default="0x0",
        description="Memory address being accessed (hex string)")
    current_cache_result: CacheLevel = Field(
        default=CacheLevel.MISS,
        description="Where this access resolved")
    is_write: bool = Field(default=False)
    current_workload: str = Field(
        default="default",
        description="Which workload generated this access")
    current_phase: str = Field(
        default="unknown",
        description="Current phase within the workload (e.g., 'forward_pass')")

    # ── PATTERN CONTEXT ──────────────────────────────────────────────────
    recent_deltas: List[int] = Field(
        default_factory=list,
        description="Last 8 address deltas in cache-line units")
    recent_pcs: List[str] = Field(
        default_factory=list,
        description="Last 8 program counters (hex)")
    recent_cache_results: List[str] = Field(
        default_factory=list,
        description="Last 8 cache results")
    pattern_analysis: PatternAnalysis = Field(
        default_factory=PatternAnalysis)

    # ── PERFORMANCE FEEDBACK ─────────────────────────────────────────────
    last_action_outcome: str = Field(
        default="",
        description="Pythia classification of last action (RAT/WASTED/...)")
    last_action_explanation: str = Field(
        default="",
        description="Why the last action got its outcome")
    prefetch_accuracy_recent: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Accuracy over last 10 prefetches")
    prefetch_accuracy_total: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Accuracy over entire episode")

    # ── SYSTEM STATE ─────────────────────────────────────────────────────
    cache_stats: CacheStats = Field(default_factory=CacheStats)
    bandwidth: BandwidthState = Field(default_factory=BandwidthState)
    mshr: MSHRState = Field(default_factory=MSHRState)
    pollution_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="pollution_evictions / total_prefetches")

    # ── MULTI-TENANT (Tasks 3-4) ─────────────────────────────────────────
    active_workloads: List[str] = Field(default_factory=list)
    workload_sla_status: Dict[str, str] = Field(
        default_factory=dict,
        description="workload_id → 'met'|'at_risk'|'violated'")
    llc_partition: Dict[str, int] = Field(
        default_factory=dict,
        description="Current LLC way allocation per workload")

    # ── AVAILABLE ACTIONS ────────────────────────────────────────────────
    available_actions: List[str] = Field(
        default_factory=list,
        description="Which ActionTypes are available in this task")
    action_hint: str = Field(
        default="",
        description="Environment hint (e.g., 'Consider reducing aggressiveness')")

    # ── EPISODE METADATA ─────────────────────────────────────────────────
    task_id: str = Field(default="")
    task_description: str = Field(default="")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0,
                          description="Normalized scalar reward [0, 1]")
    reward_detail: Reward = Field(default_factory=Reward,
                                  description="Rich reward breakdown (outcome, components, explanation)")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra data: forensics, baseline comparison, etc.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATE MODEL                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SiliconMindState(OEState):
    """Internal episode state returned by the /state endpoint.

    This is NOT exposed to the agent during normal play — it is used
    by the OpenEnv framework and for debugging.
    """

    task_id: str = Field(default="")
    step_number: int = Field(default=0, ge=0)
    max_steps: int = Field(default=0, ge=0)
    done: bool = Field(default=False)
    total_reward: float = Field(default=0.0)
    seed: int = Field(default=0)

    # Aggregate counters
    cache_stats: CacheStats = Field(default_factory=CacheStats)
    bandwidth: BandwidthState = Field(default_factory=BandwidthState)
    mshr: MSHRState = Field(default_factory=MSHRState)

    # Agent config state
    prefetch_aggressiveness: int = Field(default=2, ge=1, le=4)
    bandwidth_throttle: float = Field(default=1.0, ge=0.0, le=1.0)
    llc_partition: Dict[str, int] = Field(default_factory=dict)

    # Workload tracking
    current_workload: str = Field(default="")
    current_phase: str = Field(default="")
