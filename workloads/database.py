"""DatabaseEngine workload — B-tree queries, sequential scans, hash joins.

Archetype: PostgreSQL / MySQL query execution
SPEC CPU analog: 605.mcf_s (graph-based), 620.omnetpp_s (pointer-heavy)

Phases:
  1. B-tree Descent: Logarithmic pointer-chasing jumps (UNPREDICTABLE)
  2. Sequential Leaf Scan: Stride-1 sequential read (VERY PREDICTABLE)
  3. Hash Join Probe: hash(key) % table_size → clustered random (UNPREDICTABLE)
  4. Result Sort: Merge sort with alternating sequential streams (PREDICTABLE)
"""

from __future__ import annotations

import random
from typing import List

from workloads.base import MemoryAccess, WorkloadProfile


# Base addresses for each phase (16 MB regions)
_BASE_BTREE = 0x0010_0000
_BASE_LEAF = 0x0020_0000
_BASE_HASH = 0x0030_0000
_BASE_SORT = 0x0040_0000

# Simulated B-tree parameters
_BTREE_FANOUT = 256
_BTREE_DEPTH = 5
_BTREE_NODE_SIZE = 4096  # bytes per node (one page)

# Hash table parameters
_HASH_TABLE_SLOTS = 8192  # number of slots
_HASH_CLUSTER_SIZE = 4    # cache lines per slot cluster

# Program counters (one per phase, simulating different code paths)
_PC_BTREE = 0x0040_1000
_PC_LEAF = 0x0040_2000
_PC_HASH = 0x0040_3000
_PC_SORT = 0x0040_4000

LINE_SIZE = 64


class DatabaseEngine(WorkloadProfile):
    """B-tree + leaf scan + hash join + merge sort."""

    workload_id = "database"
    description = (
        "Database query execution: B-tree index lookup → sequential leaf scan "
        "→ hash join probe → merge sort of results"
    )

    def phases(self) -> List[str]:
        return ["btree_descent", "leaf_scan", "hash_join", "result_sort"]

    def phase_boundaries(self, length: int) -> List[tuple[int, int, str]]:
        """Allocate steps proportionally across 4 phases."""
        b = int(length * 0.12)   # ~12% B-tree (short but painful)
        l = int(length * 0.35)   # ~35% leaf scan
        h = int(length * 0.33)   # ~33% hash join
        s = length - b - l - h   # ~20% sort (remainder)
        return [
            (0, b, "btree_descent"),
            (b, b + l, "leaf_scan"),
            (b + l, b + l + h, "hash_join"),
            (b + l + h, length, "result_sort"),
        ]

    def generate_trace(self, seed: int, length: int) -> List[MemoryAccess]:
        rng = random.Random(seed)
        trace: List[MemoryAccess] = []
        boundaries = self.phase_boundaries(length)

        for start, end, phase in boundaries:
            count = end - start
            if phase == "btree_descent":
                trace.extend(self._gen_btree(rng, count))
            elif phase == "leaf_scan":
                trace.extend(self._gen_leaf_scan(rng, count))
            elif phase == "hash_join":
                trace.extend(self._gen_hash_join(rng, count))
            elif phase == "result_sort":
                trace.extend(self._gen_sort(rng, count))

        return trace[:length]

    def _gen_btree(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """B-tree descent: logarithmic jumps through tree nodes.

        Pattern: at each level, jump to a child node that is
        ~(node_size * fanout^level) bytes away.  Essentially pointer chasing.
        """
        accesses = []
        for _ in range(count):
            # Simulate one tree traversal from root to leaf
            addr = _BASE_BTREE
            level_offset = rng.randint(0, _BTREE_FANOUT - 1)
            addr += level_offset * _BTREE_NODE_SIZE
            # Add some noise to simulate different query keys
            addr += rng.randint(0, 15) * LINE_SIZE
            addr = (addr // LINE_SIZE) * LINE_SIZE  # align

            accesses.append(MemoryAccess(
                pc=_PC_BTREE,
                address=addr,
                workload_id=self.workload_id,
                phase="btree_descent",
                predictable=False,   # Pointer chasing is unpredictable
                expected_stride=0,
            ))
        return accesses

    def _gen_leaf_scan(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Sequential leaf node scan: stride-1 through sorted data.

        This is the ideal case for a nextline prefetcher.
        """
        accesses = []
        addr = _BASE_LEAF + rng.randint(0, 255) * LINE_SIZE
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_LEAF,
                address=addr,
                workload_id=self.workload_id,
                phase="leaf_scan",
                predictable=True,
                expected_stride=1,  # stride-1 in cache lines
            ))
            addr += LINE_SIZE  # next cache line
        return accesses

    def _gen_hash_join(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Hash join probe: hash(key) → random slot in hash table.

        Each probe accesses a cluster of 4 cache lines (for collision chain).
        Pattern is fundamentally UNPREDICTABLE — no stride or regularity.
        """
        accesses = []
        for _ in range(count):
            slot = rng.randint(0, _HASH_TABLE_SLOTS - 1)
            # Each slot is a cluster of _HASH_CLUSTER_SIZE cache lines
            cluster_base = _BASE_HASH + slot * _HASH_CLUSTER_SIZE * LINE_SIZE
            # Access random line within cluster
            offset = rng.randint(0, _HASH_CLUSTER_SIZE - 1) * LINE_SIZE
            addr = cluster_base + offset

            accesses.append(MemoryAccess(
                pc=_PC_HASH,
                address=addr,
                workload_id=self.workload_id,
                phase="hash_join",
                predictable=False,  # Hash lookups are unpredictable
                expected_stride=0,
            ))
        return accesses

    def _gen_sort(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Merge sort: two alternating sequential streams.

        Pattern: read from stream A (stride-1), then stream B (stride-1),
        alternating every 4-8 accesses.  The stride within each stream
        is predictable, but the switching is semi-random.
        """
        accesses = []
        stream_a = _BASE_SORT
        stream_b = _BASE_SORT + 0x10_0000  # 1MB offset
        current_stream = stream_a
        run_length = rng.randint(4, 8)
        in_run = 0

        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_SORT,
                address=current_stream,
                workload_id=self.workload_id,
                phase="result_sort",
                predictable=True,  # Within a run, it's sequential
                expected_stride=1,
            ))
            current_stream += LINE_SIZE
            in_run += 1

            if in_run >= run_length:
                # Switch streams
                if current_stream >= _BASE_SORT + 0x10_0000:
                    current_stream = stream_a
                else:
                    current_stream = stream_b
                in_run = 0
                run_length = rng.randint(4, 8)

        return accesses
