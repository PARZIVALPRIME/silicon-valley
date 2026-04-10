"""Cache hierarchy simulator — L1 / L2 / LLC with LRU and pollution tracking.

Hardware parameters match the CRL-Pythia paper (Table 5):
  L1 Data:  32 KB, 8-way,  64 B lines,  4-cycle latency
  L2:      256 KB, 8-way,  64 B lines, 12-cycle latency
  LLC:       2 MB, 16-way, 64 B lines, 40-cycle latency
  DRAM:                                200-cycle latency

Key features beyond a textbook cache simulator:
  • prefetch_bit per line — tracks whether data was brought by prefetch
  • Victim Buffer (32 entries) — detects pollution (prefetch evicting useful data)
  • Per-line access_count — enables LRU and reuse distance tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

LINE_SIZE = 64  # bytes per cache line

@dataclass
class CacheConfig:
    """Hardware parameters for one cache level."""

    name: str = "L1"
    size_bytes: int = 32_768       # 32 KB
    associativity: int = 8         # 8-way
    line_size: int = LINE_SIZE     # 64 B
    latency_cycles: int = 4

    @property
    def num_sets(self) -> int:
        return self.size_bytes // (self.associativity * self.line_size)


# Pre-built configs matching CRL-Pythia Table 5
L1_CONFIG = CacheConfig(name="L1", size_bytes=32_768,    associativity=8,  latency_cycles=4)
L2_CONFIG = CacheConfig(name="L2", size_bytes=262_144,   associativity=8,  latency_cycles=12)
LLC_CONFIG = CacheConfig(name="LLC", size_bytes=2_097_152, associativity=16, latency_cycles=40)

DRAM_LATENCY = 200  # cycles


# ───────────────────────────────────────────────────────────────────────────
# Cache Line
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class CacheLine:
    """One entry in a cache set."""

    tag: int = -1
    valid: bool = False
    dirty: bool = False
    prefetch_bit: bool = False          # True if inserted by prefetch (not demand)
    last_access_cycle: int = 0          # for LRU ordering
    access_count: int = 0              # times accessed since insertion
    inserted_by_core: int = 0          # core ID (for cross-core pollution)
    inserted_by_workload: str = ""     # workload ID


# ───────────────────────────────────────────────────────────────────────────
# Cache Set — N-way set-associative with LRU
# ───────────────────────────────────────────────────────────────────────────

class CacheSet:
    """One set in a set-associative cache.

    Replacement policy: LRU (evict the way with smallest last_access_cycle).
    """

    __slots__ = ("ways", "lines")

    def __init__(self, ways: int) -> None:
        self.ways = ways
        self.lines: List[CacheLine] = [CacheLine() for _ in range(ways)]

    def lookup(self, tag: int) -> Optional[int]:
        """Return way index if tag is present and valid, else None."""
        for i, line in enumerate(self.lines):
            if line.valid and line.tag == tag:
                return i
        return None

    def find_lru_way(self) -> int:
        """Return the index of the LRU (least-recently-used) way."""
        lru_idx = 0
        lru_cycle = self.lines[0].last_access_cycle
        for i in range(1, self.ways):
            if not self.lines[i].valid:
                return i  # invalid line → free slot
            if self.lines[i].last_access_cycle < lru_cycle:
                lru_cycle = self.lines[i].last_access_cycle
                lru_idx = i
        return lru_idx

    def insert(
        self,
        tag: int,
        cycle: int,
        is_prefetch: bool = False,
        core_id: int = 0,
        workload_id: str = "",
    ) -> Optional[CacheLine]:
        """Insert a line, evicting LRU if necessary.

        Returns the evicted CacheLine (if any), or None if a free slot was used.
        """
        way = self.lookup(tag)
        if way is not None:
            # Already present — just update access time
            line = self.lines[way]
            line.last_access_cycle = cycle
            line.access_count += 1
            if not is_prefetch:
                line.prefetch_bit = False  # demand access clears prefetch bit
            return None

        # Need to insert — find LRU victim
        lru_way = self.find_lru_way()
        victim = self.lines[lru_way]

        # Copy victim before overwriting (only if it was valid)
        evicted: Optional[CacheLine] = None
        if victim.valid:
            evicted = CacheLine(
                tag=victim.tag,
                valid=victim.valid,
                dirty=victim.dirty,
                prefetch_bit=victim.prefetch_bit,
                last_access_cycle=victim.last_access_cycle,
                access_count=victim.access_count,
                inserted_by_core=victim.inserted_by_core,
                inserted_by_workload=victim.inserted_by_workload,
            )

        # Write new line
        victim.tag = tag
        victim.valid = True
        victim.dirty = False
        victim.prefetch_bit = is_prefetch
        victim.last_access_cycle = cycle
        victim.access_count = 0
        victim.inserted_by_core = core_id
        victim.inserted_by_workload = workload_id

        return evicted

    def touch(self, tag: int, cycle: int) -> bool:
        """Update access time for a tag if present. Returns True if found."""
        way = self.lookup(tag)
        if way is not None:
            self.lines[way].last_access_cycle = cycle
            self.lines[way].access_count += 1
            self.lines[way].prefetch_bit = False  # demand clears prefetch bit
            return True
        return False


# ───────────────────────────────────────────────────────────────────────────
# Cache Level — complete cache (L1, L2, or LLC)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class AccessResult:
    """Result of accessing one level of cache."""

    hit: bool = False
    evicted: Optional[CacheLine] = None
    evicted_was_useful: bool = False    # True → pollution
    latency: int = 0


class CacheLevel:
    """A single level of cache (L1, L2, or LLC).

    Internally organized as `num_sets` CacheSets, each `associativity`-way.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.num_sets = config.num_sets
        self.sets: List[CacheSet] = [
            CacheSet(config.associativity) for _ in range(self.num_sets)
        ]

    def _set_index(self, addr: int) -> int:
        """Compute which set an address maps to."""
        block_addr = addr // self.config.line_size
        return block_addr % self.num_sets

    def _tag(self, addr: int) -> int:
        """Compute the tag for an address."""
        block_addr = addr // self.config.line_size
        return block_addr // self.num_sets

    def lookup(self, addr: int) -> bool:
        """Check if address is in cache (does NOT update LRU)."""
        set_idx = self._set_index(addr)
        tag = self._tag(addr)
        return self.sets[set_idx].lookup(tag) is not None

    def access(
        self,
        addr: int,
        cycle: int,
        is_prefetch: bool = False,
        core_id: int = 0,
        workload_id: str = "",
    ) -> AccessResult:
        """Access the cache: lookup → hit or insert (with possible eviction).

        Returns AccessResult with hit status and any evicted line.
        """
        set_idx = self._set_index(addr)
        tag = self._tag(addr)
        cache_set = self.sets[set_idx]

        # Check for hit
        way = cache_set.lookup(tag)
        if way is not None:
            cache_set.touch(tag, cycle)
            return AccessResult(
                hit=True,
                latency=self.config.latency_cycles,
            )

        # Miss — insert and possibly evict
        evicted = cache_set.insert(
            tag=tag,
            cycle=cycle,
            is_prefetch=is_prefetch,
            core_id=core_id,
            workload_id=workload_id,
        )

        return AccessResult(
            hit=False,
            evicted=evicted,
            latency=self.config.latency_cycles,
        )


# ───────────────────────────────────────────────────────────────────────────
# Victim Buffer — tracks recently evicted lines for pollution detection
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class VictimEntry:
    """A recently evicted cache line tracked for pollution detection."""

    tag: int
    set_index: int
    eviction_cycle: int
    was_prefetch_victim: bool  # True if evicted BY a prefetch insert
    original_core: int = 0
    original_workload: str = ""


class VictimBuffer:
    """FIFO buffer of recently evicted lines.

    When a demand access matches a victim, it means a prefetch evicted
    data that was actually needed → POLLUTION detected.
    """

    def __init__(self, capacity: int = 32, expiry_cycles: int = 200) -> None:
        self.capacity = capacity
        self.expiry_cycles = expiry_cycles
        self._entries: List[VictimEntry] = []

    def record_eviction(
        self,
        tag: int,
        set_index: int,
        cycle: int,
        was_prefetch_victim: bool,
        core_id: int = 0,
        workload_id: str = "",
    ) -> None:
        """Record a new eviction victim."""
        self._entries.append(VictimEntry(
            tag=tag,
            set_index=set_index,
            eviction_cycle=cycle,
            was_prefetch_victim=was_prefetch_victim,
            original_core=core_id,
            original_workload=workload_id,
        ))
        # Maintain capacity
        if len(self._entries) > self.capacity:
            self._entries.pop(0)

    def check_pollution(
        self, tag: int, set_index: int, cycle: int
    ) -> Optional[VictimEntry]:
        """Check if a demand access matches a recently evicted victim.

        If found, it indicates pollution — a prefetch evicted this useful line.
        Returns the VictimEntry if pollution, else None.
        """
        for i, entry in enumerate(self._entries):
            if (
                entry.tag == tag
                and entry.set_index == set_index
                and entry.was_prefetch_victim
                and (cycle - entry.eviction_cycle) <= self.expiry_cycles
            ):
                # Found pollution — remove from buffer
                self._entries.pop(i)
                return entry
        return None

    def expire(self, cycle: int) -> None:
        """Remove entries older than expiry_cycles."""
        self._entries = [
            e for e in self._entries
            if (cycle - e.eviction_cycle) <= self.expiry_cycles
        ]


# ───────────────────────────────────────────────────────────────────────────
# Cache Hierarchy — L1 → L2 → LLC → Memory (the full pipeline)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class HierarchyAccessResult:
    """Full result of a memory access through the entire hierarchy."""

    hit: bool = False
    hit_level: str = "MISS"             # "L1_HIT" | "L2_HIT" | "LLC_HIT" | "MISS"
    total_latency: int = 0
    pollution_detected: bool = False
    pollution_victim: Optional[VictimEntry] = None
    evicted_from_llc: Optional[CacheLine] = None


@dataclass
class PrefetchResult:
    """Result of issuing a prefetch request."""

    issued: bool = True                 # False if redundant (already in cache)
    redundant: bool = False
    evicted: Optional[CacheLine] = None
    evicted_was_prefetch: bool = False  # The evicted line was also a prefetch
    pollution_risk: bool = False        # Evicted a demand-accessed line


class CacheHierarchy:
    """Complete L1 → L2 → LLC → DRAM cache hierarchy.

    This is the core simulation engine.  It processes demand accesses
    and prefetch insertions, tracking hits/misses/evictions at every level
    and detecting pollution through the victim buffer.
    """

    def __init__(
        self,
        l1_config: CacheConfig = L1_CONFIG,
        l2_config: Optional[CacheConfig] = L2_CONFIG,
        llc_config: Optional[CacheConfig] = LLC_CONFIG,
        dram_latency: int = DRAM_LATENCY,
        victim_buffer_size: int = 32,
        victim_expiry_cycles: int = 200,
    ) -> None:
        self.l1 = CacheLevel(l1_config)
        self.l2 = CacheLevel(l2_config) if l2_config else None
        self.llc = CacheLevel(llc_config) if llc_config else None
        self.dram_latency = dram_latency
        self.victim_buffer = VictimBuffer(victim_buffer_size, victim_expiry_cycles)

        # Counters
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.llc_hits = 0
        self.llc_misses = 0
        self.total_accesses = 0
        self.pollution_events = 0
        self.cross_core_pollution = 0

    def access(
        self,
        addr: int,
        cycle: int,
        core_id: int = 0,
        workload_id: str = "",
    ) -> HierarchyAccessResult:
        """Process a demand memory access through the full hierarchy.

        Flow: L1 → (miss) → L2 → (miss) → LLC → (miss) → DRAM
        On miss at each level, the line is inserted into that level.
        """
        self.total_accesses += 1
        result = HierarchyAccessResult()

        # Expire old victim entries
        self.victim_buffer.expire(cycle)

        # Check for pollution: is this address in the victim buffer?
        l1_set_idx = self.l1._set_index(addr)
        l1_tag = self.l1._tag(addr)
        pollution_victim = self.victim_buffer.check_pollution(l1_tag, l1_set_idx, cycle)
        if pollution_victim is not None:
            result.pollution_detected = True
            result.pollution_victim = pollution_victim
            self.pollution_events += 1
            if pollution_victim.original_core != core_id:
                self.cross_core_pollution += 1

        # ── L1 ──
        l1_result = self.l1.access(
            addr, cycle, is_prefetch=False, core_id=core_id, workload_id=workload_id,
        )
        if l1_result.hit:
            self.l1_hits += 1
            result.hit = True
            result.hit_level = "L1_HIT"
            result.total_latency = l1_result.latency
            return result

        self.l1_misses += 1

        # Track L1 eviction in victim buffer (if evicted by this demand insert)
        if l1_result.evicted and l1_result.evicted.valid:
            self.victim_buffer.record_eviction(
                tag=l1_result.evicted.tag,
                set_index=l1_set_idx,
                cycle=cycle,
                was_prefetch_victim=False,  # evicted by demand, not prefetch
                core_id=l1_result.evicted.inserted_by_core,
                workload_id=l1_result.evicted.inserted_by_workload,
            )

        latency = l1_result.latency  # L1 miss penalty

        # ── L2 ──
        if self.l2 is not None:
            l2_result = self.l2.access(
                addr, cycle, is_prefetch=False, core_id=core_id, workload_id=workload_id,
            )
            if l2_result.hit:
                self.l2_hits += 1
                result.hit = True
                result.hit_level = "L2_HIT"
                result.total_latency = latency + l2_result.latency
                return result
            self.l2_misses += 1
            latency += l2_result.latency

        # ── LLC ──
        if self.llc is not None:
            llc_result = self.llc.access(
                addr, cycle, is_prefetch=False, core_id=core_id, workload_id=workload_id,
            )
            if llc_result.hit:
                self.llc_hits += 1
                result.hit = True
                result.hit_level = "LLC_HIT"
                result.total_latency = latency + llc_result.latency
                return result
            self.llc_misses += 1
            latency += llc_result.latency
            result.evicted_from_llc = llc_result.evicted

        # ── DRAM ──
        result.hit = False
        result.hit_level = "MISS"
        result.total_latency = latency + self.dram_latency
        return result

    def prefetch(
        self,
        addr: int,
        cycle: int,
        core_id: int = 0,
        workload_id: str = "",
    ) -> PrefetchResult:
        """Insert a prefetched line into the cache hierarchy.

        The line is inserted into L1 (like a real hardware prefetch fill).
        If it's already present anywhere in the hierarchy → redundant.
        """
        # Check if already in L1 → redundant
        if self.l1.lookup(addr):
            return PrefetchResult(issued=False, redundant=True)

        # Check L2 / LLC — if found, promote to L1
        found_in_lower = False
        if self.l2 and self.l2.lookup(addr):
            found_in_lower = True
        elif self.llc and self.llc.lookup(addr):
            found_in_lower = True

        if found_in_lower:
            return PrefetchResult(issued=False, redundant=True)

        # Not in any cache level — issue the prefetch into L1
        l1_set_idx = self.l1._set_index(addr)
        l1_result = self.l1.access(
            addr, cycle, is_prefetch=True, core_id=core_id, workload_id=workload_id,
        )

        evicted = l1_result.evicted
        pollution_risk = False

        if evicted and evicted.valid:
            # Record in victim buffer — this eviction was caused by a PREFETCH
            self.victim_buffer.record_eviction(
                tag=evicted.tag,
                set_index=l1_set_idx,
                cycle=cycle,
                was_prefetch_victim=True,  # evicted BY a prefetch
                core_id=evicted.inserted_by_core,
                workload_id=evicted.inserted_by_workload,
            )
            # If evicted line had been accessed (prefetch_bit cleared), it was useful
            if not evicted.prefetch_bit and evicted.access_count > 0:
                pollution_risk = True

        # Also insert into LLC for inclusive behavior
        if self.llc is not None:
            self.llc.access(
                addr, cycle, is_prefetch=True, core_id=core_id, workload_id=workload_id,
            )

        return PrefetchResult(
            issued=True,
            redundant=False,
            evicted=evicted,
            evicted_was_prefetch=evicted.prefetch_bit if evicted else False,
            pollution_risk=pollution_risk,
        )

    def reset(self) -> None:
        """Reset all cache state and counters."""
        self.l1 = CacheLevel(self.l1.config)
        if self.l2:
            self.l2 = CacheLevel(self.l2.config)
        if self.llc:
            self.llc = CacheLevel(self.llc.config)
        self.victim_buffer = VictimBuffer(
            self.victim_buffer.capacity, self.victim_buffer.expiry_cycles,
        )
        self.l1_hits = self.l1_misses = 0
        self.l2_hits = self.l2_misses = 0
        self.llc_hits = self.llc_misses = 0
        self.total_accesses = 0
        self.pollution_events = 0
        self.cross_core_pollution = 0

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        total_hits = self.l1_hits + self.l2_hits + self.llc_hits
        return total_hits / self.total_accesses

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def llc_hit_rate(self) -> float:
        total = self.llc_hits + self.llc_misses
        return self.llc_hits / total if total > 0 else 0.0
