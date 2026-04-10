"""Task 1: Pattern Scout (Easy)

"Can the agent identify simple, regular patterns and leave unpredictable ones alone?"

Trace: DatabaseEngine leaf scan + sort phases ONLY (predictable sequential).
Cache: L1 only.  Bandwidth: unlimited.  MSHR: unlimited.
Actions: PREFETCH and NO_PREFETCH only.

Expected baseline score: 0.40-0.60
"""

from __future__ import annotations

from typing import Dict, List

from engine.cache import CacheConfig, L1_CONFIG
from engine.bandwidth import BandwidthConfig
from engine.mshr import MSHRConfig
from models import ActionType
from tasks.base import BaseTask
from workloads.base import MemoryAccess
from workloads.database import DatabaseEngine


class PatternScoutTask(BaseTask):
    """Easy task: learn stride-1 and stride-4, avoid prefetching noise."""

    task_id = "easy_pattern_scout"
    title = "Pattern Scout"
    description = (
        "Sequential and stride patterns from a database leaf scan. "
        "Single cache level (L1 only). No bandwidth pressure. "
        "Learn to prefetch obvious patterns and hold back on noise."
    )
    difficulty = "easy"
    max_steps = 140

    def build_trace(self, seed: int) -> List[MemoryAccess]:
        """Generate trace: leaf scan + sort phases from DatabaseEngine.

        Composition:
          - Steps 1-98 (~70%): Pure sequential leaf scan (stride-1)
          - Steps 99-126 (~20%): Merge sort with alternating streams
          - Steps 127-140 (~10%): Random noise (unpredictable)

        The last 10% is random to test restraint (NO_PREFETCH).
        """
        db = DatabaseEngine()
        # Generate a full DB trace and extract just the easy parts
        full_trace = db.generate_trace(seed, 250)

        trace: List[MemoryAccess] = []

        # Extract leaf scan accesses (sequential stride-1)
        leaf_accesses = [a for a in full_trace if a.phase == "leaf_scan"][:98]
        trace.extend(leaf_accesses)

        # Extract sort accesses (alternating sequential)
        sort_accesses = [a for a in full_trace if a.phase == "result_sort"][:28]
        trace.extend(sort_accesses)

        # Add random noise for restraint testing
        import random as rng_module
        rng = rng_module.Random(seed + 999)
        for i in range(14):
            trace.append(MemoryAccess(
                pc=0x0040_9000,
                address=rng.randint(0x0050_0000, 0x005F_FFFF) & ~63,  # Random, aligned
                workload_id="database",
                phase="noise",
                predictable=False,
                expected_stride=0,
            ))

        return trace[:self.max_steps]

    def cache_configs(self) -> List[CacheConfig]:
        """L1 only — simplest cache configuration."""
        return [L1_CONFIG]

    def bandwidth_config(self) -> BandwidthConfig:
        """Unlimited bandwidth (no contention)."""
        return BandwidthConfig(max_outstanding=64, penalty_threshold=0.99)

    def mshr_config(self) -> MSHRConfig:
        """32 MSHRs (effectively unlimited)."""
        return MSHRConfig(capacity=32)

    def available_actions(self) -> List[ActionType]:
        return [ActionType.PREFETCH, ActionType.NO_PREFETCH]

    def grader_weights(self) -> Dict[str, float]:
        return {
            "accuracy": 0.35,
            "coverage": 0.30,
            "restraint": 0.20,
            "timeliness": 0.15,
        }

    def workload_ids(self) -> List[str]:
        return ["database"]
