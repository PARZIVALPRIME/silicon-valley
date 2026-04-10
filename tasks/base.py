"""Task base class and configuration dataclasses.

Each task defines:
  - Trace composition (which workloads, how many steps)
  - Hardware configuration (which cache levels, MSHR capacity, BW limits)
  - Available actions (easy tasks only have prefetch/no_prefetch)
  - Grader weights (what metrics matter most)
  - SLA requirements (multi-tenant tasks)
  - Regime changes (when workload patterns change mid-episode)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from engine.cache import CacheConfig, L1_CONFIG, L2_CONFIG, LLC_CONFIG
from engine.bandwidth import BandwidthConfig
from engine.mshr import MSHRConfig
from models import ActionType
from workloads.base import MemoryAccess


@dataclass
class SLARequirement:
    """Service-level agreement for a workload in multi-tenant tasks."""

    workload_id: str
    priority: str = "MEDIUM"         # HIGH | MEDIUM | LOW
    max_avg_latency: float = 100.0   # cycles
    min_hit_rate: float = 0.50
    description: str = ""


class BaseTask(ABC):
    """Abstract base class for all SiliconMind tasks.

    Subclasses configure:
      - Hardware parameters (cache, BW, MSHR)
      - Trace generation (workloads, length, composition)
      - Grading weights (what matters in this task)
      - Available actions (progressive unlock)
    """

    task_id: str = ""
    title: str = ""
    description: str = ""
    difficulty: str = "easy"        # easy | medium | hard | extreme
    max_steps: int = 200

    # ── Abstract methods ─────────────────────────────────────────────────

    @abstractmethod
    def build_trace(self, seed: int) -> List[MemoryAccess]:
        """Generate the complete memory access trace for this task."""
        ...

    @abstractmethod
    def cache_configs(self) -> List[CacheConfig]:
        """Return cache level configurations (L1 only, L1+L2, L1+L2+LLC)."""
        ...

    @abstractmethod
    def grader_weights(self) -> Dict[str, float]:
        """Return {metric_name: weight} for final scoring."""
        ...

    # ── Default implementations (override as needed) ─────────────────────

    def bandwidth_config(self) -> BandwidthConfig:
        """Default: unlimited bandwidth (easy tasks)."""
        return BandwidthConfig(max_outstanding=64, penalty_threshold=0.95)

    def mshr_config(self) -> MSHRConfig:
        """Default: 32 MSHRs (effectively unlimited for easy tasks)."""
        return MSHRConfig(capacity=32)

    def available_actions(self) -> List[ActionType]:
        """Default: only prefetch and no_prefetch."""
        return [ActionType.PREFETCH, ActionType.NO_PREFETCH]

    def sla_requirements(self) -> List[SLARequirement]:
        """Default: no SLAs (single-tenant tasks)."""
        return []

    def regime_changes(self) -> Dict[int, str]:
        """Default: no regime changes. Override for tasks with mid-episode changes.

        Returns {step_number: description_of_change}.
        """
        return {}

    def workload_ids(self) -> List[str]:
        """Return list of workload IDs in this task."""
        return ["default"]
