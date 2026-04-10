"""Baseline counterfactual engine — runs traces with no prefetching.

During environment reset(), we run the same trace through a no-prefetch
cache hierarchy to establish baseline miss rates.  This enables:

  coverage = (baseline_misses - agent_misses) / baseline_misses

This is the industry-standard way to evaluate prefetchers:
"How many misses did you eliminate compared to doing nothing?"
"""

from __future__ import annotations

from typing import List

from engine.cache import CacheConfig, CacheHierarchy
from workloads.base import MemoryAccess


def compute_baseline_misses(
    trace: List[MemoryAccess],
    cache_configs: List[CacheConfig],
) -> int:
    """Run the trace through the cache hierarchy with ZERO prefetching.

    Returns the total number of demand misses (across all cache levels).
    This is the baseline that the agent's coverage is measured against.
    """
    # Build cache hierarchy from configs
    l1_config = cache_configs[0] if len(cache_configs) > 0 else None
    l2_config = cache_configs[1] if len(cache_configs) > 1 else None
    llc_config = cache_configs[2] if len(cache_configs) > 2 else None

    if l1_config is None:
        return len(trace)  # no cache = all misses

    hierarchy = CacheHierarchy(
        l1_config=l1_config,
        l2_config=l2_config,
        llc_config=llc_config,
    )

    total_misses = 0
    for i, access in enumerate(trace):
        result = hierarchy.access(
            addr=access.address,
            cycle=i * 10,  # approximate cycle counter
            workload_id=access.workload_id,
        )
        if not result.hit:
            total_misses += 1

    return total_misses
