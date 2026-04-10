"""Task 2: Workload Whisperer (Medium)

"Can the agent detect phase transitions and adapt its strategy?"

Trace: Full MLTrainingLoop (all 5 phases with transitions).
Cache: L1 + L2.  Bandwidth: limited.  MSHR: 16.
Actions: PREFETCH, NO_PREFETCH, SET_AGGRESSIVENESS.

The killer: forward→backward phase transition inverts the access pattern.
Agent must detect accuracy crash and adapt.

Expected baseline score: 0.25-0.40
"""

from __future__ import annotations

from typing import Dict, List

from engine.cache import CacheConfig, L1_CONFIG, L2_CONFIG
from engine.bandwidth import BandwidthConfig
from engine.mshr import MSHRConfig
from models import ActionType
from tasks.base import BaseTask
from workloads.base import MemoryAccess
from workloads.ml_training import MLTrainingLoop


class WorkloadWhispererTask(BaseTask):
    """Medium task: phase-changing ML training with bandwidth pressure."""

    task_id = "medium_workload_whisperer"
    title = "Workload Whisperer"
    description = (
        "ML training loop with 5 phases and dramatic pattern changes. "
        "Two cache levels (L1+L2). Bandwidth pressure above 70%. "
        "The forward→backward transition inverts the access pattern — "
        "your stride-1 strategy will fail and you must adapt."
    )
    difficulty = "medium"
    max_steps = 500

    def build_trace(self, seed: int) -> List[MemoryAccess]:
        """Full ML training loop: forward → activation → backward → gradient → optimizer."""
        ml = MLTrainingLoop()
        return ml.generate_trace(seed, self.max_steps)

    def cache_configs(self) -> List[CacheConfig]:
        """L1 + L2 (no LLC)."""
        return [L1_CONFIG, L2_CONFIG]

    def bandwidth_config(self) -> BandwidthConfig:
        """Limited bandwidth: penalty above 70%, configured for medium pressure."""
        return BandwidthConfig(
            max_outstanding=32,
            penalty_threshold=0.70,
            drop_threshold=0.95,
        )

    def mshr_config(self) -> MSHRConfig:
        """16 MSHRs — realistic constraint."""
        return MSHRConfig(capacity=16)

    def available_actions(self) -> List[ActionType]:
        return [
            ActionType.PREFETCH,
            ActionType.NO_PREFETCH,
            ActionType.SET_AGGRESSIVENESS,
        ]

    def regime_changes(self) -> Dict[int, str]:
        """Phase transitions in ML training."""
        boundaries = MLTrainingLoop().phase_boundaries(self.max_steps)
        changes = {}
        for start, _, phase in boundaries[1:]:  # skip first phase
            changes[start] = f"Phase transition to {phase}"
        return changes

    def grader_weights(self) -> Dict[str, float]:
        return {
            "accuracy": 0.20,
            "coverage": 0.20,
            "pollution": 0.20,
            "bandwidth": 0.15,
            "adaptation": 0.15,
            "restraint": 0.10,
        }

    def workload_ids(self) -> List[str]:
        return ["ml_training"]
