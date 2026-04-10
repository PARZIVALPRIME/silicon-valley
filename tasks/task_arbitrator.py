"""Task 3: Noisy Neighbor Arbitrator (Hard)

"Can the agent manage shared resources fairly across competing workloads?"

Trace: 3 interleaved workloads (web + database + video) competing for:
  - Shared LLC (2MB, 16-way)
  - Shared memory bandwidth
  - Shared MSHRs

VideoEncoder's reference frame loading can saturate bandwidth and
thrash the LLC, destroying WebServer's latency SLA.

Actions: ALL 5 types including cache partitioning and bandwidth throttling.

Expected baseline score: 0.10-0.25
"""

from __future__ import annotations

from typing import Dict, List

from engine.cache import CacheConfig, L1_CONFIG, L2_CONFIG, LLC_CONFIG
from engine.bandwidth import BandwidthConfig
from engine.mshr import MSHRConfig
from models import ActionType
from tasks.base import BaseTask, SLARequirement
from workloads.base import MemoryAccess
from workloads.composer import WorkloadComposer
from workloads.database import DatabaseEngine
from workloads.web_server import WebServer
from workloads.video_encoder import VideoEncoder


class NoisyNeighborTask(BaseTask):
    """Hard task: multi-tenant contention with SLA requirements."""

    task_id = "hard_noisy_neighbor"
    title = "Noisy Neighbor Arbitrator"
    description = (
        "3 competing workloads sharing LLC and memory bandwidth. "
        "WebServer needs low latency, DatabaseEngine needs throughput, "
        "VideoEncoder is bandwidth-hungry. "
        "Your job: partition cache, throttle bandwidth, and ensure "
        "ALL workload SLAs are met — not just the easy ones."
    )
    difficulty = "hard"
    max_steps = 750

    def build_trace(self, seed: int) -> List[MemoryAccess]:
        """Interleave web + database + video, 250 accesses each."""
        composer = WorkloadComposer()
        composer.add(WebServer(), trace_length=250)
        composer.add(DatabaseEngine(), trace_length=250)
        composer.add(VideoEncoder(), trace_length=250)
        return composer.compose(seed)[:self.max_steps]

    def cache_configs(self) -> List[CacheConfig]:
        """Full hierarchy: L1 + L2 + shared LLC."""
        return [L1_CONFIG, L2_CONFIG, LLC_CONFIG]

    def bandwidth_config(self) -> BandwidthConfig:
        """Severely limited: penalty above 60%, prefetch drops above 90%."""
        return BandwidthConfig(
            max_outstanding=32,
            penalty_threshold=0.60,
            drop_threshold=0.90,
        )

    def mshr_config(self) -> MSHRConfig:
        """16 MSHRs shared across all workloads."""
        return MSHRConfig(capacity=16)

    def available_actions(self) -> List[ActionType]:
        """All 5 control levers unlocked."""
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
                workload_id="web_server",
                priority="HIGH",
                max_avg_latency=50.0,
                min_hit_rate=0.75,
                description="Latency-critical. SLA violation = customer-facing impact.",
            ),
            SLARequirement(
                workload_id="database",
                priority="MEDIUM",
                max_avg_latency=100.0,
                min_hit_rate=0.55,
                description="Throughput-sensitive. Can tolerate some latency spikes.",
            ),
            SLARequirement(
                workload_id="video_encoder",
                priority="LOW",
                max_avg_latency=300.0,
                min_hit_rate=0.30,
                description="Best-effort batch processing. Can be throttled.",
            ),
        ]

    def grader_weights(self) -> Dict[str, float]:
        return {
            "sla": 0.20,
            "pollution": 0.15,
            "fairness": 0.15,
            "coverage": 0.15,
            "bandwidth": 0.10,
            "accuracy": 0.10,
            "timeliness": 0.10,
            "restraint": 0.05,
        }

    def workload_ids(self) -> List[str]:
        return ["web_server", "database", "video_encoder"]
