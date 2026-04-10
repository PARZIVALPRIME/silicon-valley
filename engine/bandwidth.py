"""Memory bandwidth contention model.

Simulates a DDR4-3200 single-channel memory bus with:
  • Peak bandwidth: 25.6 GB/s
  • Quadratic latency penalty above 70% utilization
  • Prefetch request DROPPING above 95% utilization
  • Per-workload bandwidth accounting for multi-tenant fairness

This directly affects the Pythia reward classification:
  RIN-L vs RIN-H depends on whether bandwidth is above/below threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from models import BandwidthPressure


@dataclass
class BandwidthConfig:
    """Memory bus configuration."""

    peak_gbps: float = 25.6            # DDR4-3200 single channel
    bus_width_bytes: int = 8            # 64-bit bus
    max_outstanding: int = 32           # max in-flight requests
    penalty_threshold: float = 0.70     # quadratic penalty starts here
    drop_threshold: float = 0.95        # prefetch requests dropped here
    line_transfer_cycles: int = 10      # cycles to transfer one cache line (64B)


class BandwidthController:
    """Simulates memory bus contention with quadratic back-pressure.

    Usage:
        bw = BandwidthController()
        accepted, extra_latency = bw.request(is_prefetch=True)
        bw.tick()  # advance one cycle
    """

    def __init__(self, config: BandwidthConfig = BandwidthConfig()) -> None:
        self.config = config
        self._outstanding: int = 0      # currently in-flight requests
        self._completing: list[int] = []  # cycle countdown for each request
        self._cycle: int = 0

        # Counters
        self.total_requests: int = 0
        self.total_drops: int = 0
        self.prefetch_drops: int = 0
        self._utilization_samples: list[float] = []

        # Per-workload tracking
        self.per_workload_bytes: Dict[str, int] = {}

    @property
    def utilization(self) -> float:
        """Current bandwidth utilization as a fraction [0.0, 1.0]."""
        if self.config.max_outstanding == 0:
            return 0.0
        return min(1.0, self._outstanding / self.config.max_outstanding)

    @property
    def pressure(self) -> BandwidthPressure:
        """Qualitative pressure level for the observation."""
        u = self.utilization
        if u < 0.50:
            return BandwidthPressure.LOW
        if u < 0.70:
            return BandwidthPressure.MODERATE
        if u < 0.90:
            return BandwidthPressure.HIGH
        return BandwidthPressure.CRITICAL

    @property
    def is_high_bandwidth(self) -> bool:
        """Whether bandwidth is above the Pythia high-BW threshold (70%)."""
        return self.utilization > self.config.penalty_threshold

    @property
    def avg_utilization(self) -> float:
        if not self._utilization_samples:
            return 0.0
        return sum(self._utilization_samples) / len(self._utilization_samples)

    def latency_multiplier(self) -> float:
        """Quadratic latency penalty above threshold.

        Below 70%: 1.0× (no penalty)
        At 85%: ~1.75× latency
        At 100%: ~4.0× latency
        """
        u = self.utilization
        if u <= self.config.penalty_threshold:
            return 1.0
        excess = (u - self.config.penalty_threshold) / (1.0 - self.config.penalty_threshold)
        return 1.0 + 3.0 * (excess ** 2)

    def request(
        self,
        is_prefetch: bool = False,
        workload_id: str = "",
    ) -> tuple[bool, int]:
        """Try to issue a memory bus request.

        Args:
            is_prefetch: True for prefetch, False for demand.
            workload_id: Which workload issued this request.

        Returns:
            (accepted: bool, extra_latency_cycles: int)
            If not accepted, the request was dropped (only prefetches drop).
        """
        self.total_requests += 1

        # Drop prefetches when bandwidth is critical
        if is_prefetch and self.utilization >= self.config.drop_threshold:
            self.total_drops += 1
            self.prefetch_drops += 1
            return False, 0

        # Demand requests always go through (they stall the pipeline)
        self._outstanding += 1
        transfer_time = int(self.config.line_transfer_cycles * self.latency_multiplier())
        self._completing.append(transfer_time)

        # Track per-workload bandwidth
        if workload_id:
            self.per_workload_bytes[workload_id] = (
                self.per_workload_bytes.get(workload_id, 0) + 64  # one cache line
            )

        extra_latency = max(0, transfer_time - self.config.line_transfer_cycles)
        return True, extra_latency

    def tick(self, cycles: int = 1) -> None:
        """Advance the simulation by N cycles.

        Completes any in-flight requests whose countdown reaches zero.
        """
        self._cycle += cycles
        completed = []
        remaining = []
        for countdown in self._completing:
            new_val = countdown - cycles
            if new_val <= 0:
                completed.append(new_val)
            else:
                remaining.append(new_val)
        self._completing = remaining
        self._outstanding = max(0, self._outstanding - len(completed))

        # Record utilization sample
        self._utilization_samples.append(self.utilization)

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self._outstanding = 0
        self._completing.clear()
        self._cycle = 0
        self.total_requests = 0
        self.total_drops = 0
        self.prefetch_drops = 0
        self._utilization_samples.clear()
        self.per_workload_bytes.clear()
