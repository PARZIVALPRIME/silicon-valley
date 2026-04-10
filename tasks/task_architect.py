"""Task 4: Silicon Architect (Extreme)

"Can the agent coordinate prefetching across cores, handle workload
migration, and manage under severe resource pressure?"

Trace: 2 cores interleaved — Core0=DatabaseEngine, Core1=MLTrainingLoop
       At step 500, Core1 switches from ML → VideoEncoder (regime change!)

Hardware: Per-core L1+L2, shared LLC (2MB, 16-way).
          Only 8 MSHRs per core.  Severely limited bandwidth.

Cross-core pollution: Core0's prefetch can evict Core1's useful LLC data.

Expected baseline score: 0.00-0.12
"""

from __future__ import annotations

from typing import Dict, List

from engine.cache import CacheConfig, L1_CONFIG, L2_CONFIG, LLC_CONFIG
from engine.bandwidth import BandwidthConfig
from engine.mshr import MSHRConfig
from models import ActionType
from tasks.base import BaseTask, SLARequirement
from workloads.base import MemoryAccess
from workloads.database import DatabaseEngine
from workloads.ml_training import MLTrainingLoop
from workloads.video_encoder import VideoEncoder


class SiliconArchitectTask(BaseTask):
    """Extreme task: multi-core with regime change and cross-core pollution."""

    task_id = "extreme_silicon_architect"
    title = "Silicon Architect"
    description = (
        "2-core system: Core0 runs DatabaseEngine, Core1 starts with "
        "MLTrainingLoop then switches to VideoEncoder at step 500. "
        "Shared LLC, severely limited bandwidth, only 8 MSHRs per core. "
        "Cross-core pollution is tracked with 2× penalty. "
        "The regime change invalidates your learned strategy mid-episode."
    )
    difficulty = "extreme"
    max_steps = 1000

    def build_trace(self, seed: int) -> List[MemoryAccess]:
        """Two cores interleaved, with regime change at step 500.

        Steps 1-500: Core0=DB[0], Core1=ML[0], Core0=DB[1], Core1=ML[1], ...
        Steps 501-1000: Core0=DB[250], Core1=Video[0], Core0=DB[251], ...
        """
        db = DatabaseEngine()
        ml = MLTrainingLoop()
        video = VideoEncoder()

        # Generate individual traces
        db_trace = db.generate_trace(seed, 500)
        ml_trace = ml.generate_trace(seed + 1, 250)
        video_trace = video.generate_trace(seed + 2, 250)

        # Tag with core IDs (using workload_id prefix for disambiguation)
        for access in db_trace:
            access.workload_id = "core0_database"
        for access in ml_trace:
            access.workload_id = "core1_ml_training"
        for access in video_trace:
            access.workload_id = "core1_video_encoder"

        # Interleave: core0, core1, core0, core1, ...
        trace: List[MemoryAccess] = []

        # Phase 1 (steps 0-499): DB interleaved with ML
        for i in range(250):
            if i < len(db_trace):
                trace.append(db_trace[i])
            if i < len(ml_trace):
                trace.append(ml_trace[i])

        # Phase 2 (steps 500-999): DB interleaved with Video
        for i in range(250):
            db_idx = 250 + i
            if db_idx < len(db_trace):
                trace.append(db_trace[db_idx])
            if i < len(video_trace):
                trace.append(video_trace[i])

        return trace[:self.max_steps]

    def cache_configs(self) -> List[CacheConfig]:
        """Per-core L1+L2, shared LLC."""
        return [L1_CONFIG, L2_CONFIG, LLC_CONFIG]

    def bandwidth_config(self) -> BandwidthConfig:
        """Severely limited: penalty above 55%."""
        return BandwidthConfig(
            max_outstanding=24,
            penalty_threshold=0.55,
            drop_threshold=0.90,
        )

    def mshr_config(self) -> MSHRConfig:
        """Only 8 MSHRs per core — severe constraint."""
        return MSHRConfig(capacity=8)

    def available_actions(self) -> List[ActionType]:
        """All 5 control levers."""
        return [
            ActionType.PREFETCH,
            ActionType.NO_PREFETCH,
            ActionType.SET_AGGRESSIVENESS,
            ActionType.THROTTLE_BANDWIDTH,
            ActionType.PARTITION_CACHE,
        ]

    def sla_requirements(self) -> List[SLARequirement]:
        return [
            SLARequirement(
                workload_id="core0_database",
                priority="HIGH",
                max_avg_latency=80.0,
                min_hit_rate=0.55,
                description="Database on Core0 — must maintain throughput.",
            ),
            SLARequirement(
                workload_id="core1_ml_training",
                priority="MEDIUM",
                max_avg_latency=120.0,
                min_hit_rate=0.40,
                description="ML training on Core1 — tolerates some latency.",
            ),
            SLARequirement(
                workload_id="core1_video_encoder",
                priority="LOW",
                max_avg_latency=300.0,
                min_hit_rate=0.25,
                description="Video encoder on Core1 (after regime change). Best-effort.",
            ),
        ]

    def regime_changes(self) -> Dict[int, str]:
        """Step 500: Core1 switches from ML Training → Video Encoder."""
        return {
            500: "Core1 workload switch: MLTrainingLoop → VideoEncoder",
        }

    def grader_weights(self) -> Dict[str, float]:
        return {
            "coverage": 0.12,
            "accuracy": 0.12,
            "pollution": 0.10,
            "cross_core_pollution": 0.12,
            "fairness": 0.12,
            "bandwidth": 0.10,
            "adaptation": 0.12,
            "sla": 0.10,
            "timeliness": 0.05,
            "restraint": 0.05,
        }

    def workload_ids(self) -> List[str]:
        return ["core0_database", "core1_ml_training", "core1_video_encoder"]
