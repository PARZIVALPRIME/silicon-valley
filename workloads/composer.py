"""Multi-workload composer — interleaves traces from multiple workloads.

Used by Task 3 (Noisy Neighbor) and Task 4 (Silicon Architect) to
create interleaved access streams that simulate multi-tenant contention.

Interleaving is round-robin: one access from each workload in turn.
Each workload's addresses are in isolated address space regions,
so they compete for cache/bandwidth but not for data.
"""

from __future__ import annotations

from typing import List

from workloads.base import MemoryAccess, WorkloadProfile


class WorkloadComposer:
    """Compose multiple workload traces into a single interleaved trace.

    Usage:
        composer = WorkloadComposer()
        composer.add(DatabaseEngine(), trace_length=250)
        composer.add(WebServer(), trace_length=250)
        composer.add(VideoEncoder(), trace_length=250)
        trace = composer.compose(seed=42)

    The resulting trace interleaves round-robin:
        db[0], web[0], video[0], db[1], web[1], video[1], ...
    """

    def __init__(self) -> None:
        self._workloads: List[tuple[WorkloadProfile, int]] = []

    def add(self, workload: WorkloadProfile, trace_length: int) -> None:
        """Add a workload with the number of accesses to generate."""
        self._workloads.append((workload, trace_length))

    def compose(self, seed: int) -> List[MemoryAccess]:
        """Generate and interleave all traces.

        Each workload gets a unique sub-seed for determinism.
        """
        if not self._workloads:
            return []

        # Generate individual traces with deterministic sub-seeds
        traces: List[List[MemoryAccess]] = []
        for i, (workload, length) in enumerate(self._workloads):
            sub_seed = seed * 1000 + i * 137  # deterministic, collision-free
            trace = workload.generate_trace(sub_seed, length)
            traces.append(trace)

        # Round-robin interleave
        composed: List[MemoryAccess] = []
        max_len = max(len(t) for t in traces)

        for step in range(max_len):
            for trace in traces:
                if step < len(trace):
                    composed.append(trace[step])

        return composed

    @property
    def workload_ids(self) -> List[str]:
        """List of workload IDs in order."""
        return [w.workload_id for w, _ in self._workloads]

    @property
    def total_length(self) -> int:
        """Total number of accesses in the composed trace."""
        return sum(length for _, length in self._workloads)
