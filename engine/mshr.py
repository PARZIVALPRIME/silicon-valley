"""Miss Status Holding Registers (MSHR) simulation.

MSHRs track outstanding cache misses.  Each MSHR entry holds:
  - The requested address
  - Whether it's a prefetch or demand
  - When it was allocated (for latency tracking)

Key behaviors:
  • Demand requests have PRIORITY over prefetches for MSHR slots
  • When MSHRs are full, demand requests STALL the pipeline
  • Secondary misses (same address already pending) merge into existing entry
  • Prefetches can be evicted if MSHR is full and a demand comes in
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import MSHRRisk


@dataclass
class MSHRConfig:
    """MSHR configuration per core."""

    capacity: int = 16           # Total MSHR entries
    merge_enabled: bool = True   # Allow secondary miss merging
    demand_priority: bool = True  # Demands evict prefetches when full


@dataclass
class MSHREntry:
    """One outstanding miss in the MSHR queue."""

    address: int                 # Cache-line-aligned address
    is_prefetch: bool            # True if this is a prefetch request
    cycle_entered: int           # When this entry was allocated
    expected_completion: int     # When the data should arrive
    workload_id: str = ""
    core_id: int = 0


class MSHRQueue:
    """MSHR queue with demand priority and secondary miss merging.

    Usage:
        mshr = MSHRQueue()
        success = mshr.allocate(addr, is_prefetch=True, cycle=100)
        mshr.tick(cycle=110)  # completes entries past their deadline
    """

    def __init__(self, config: MSHRConfig = MSHRConfig()) -> None:
        self.config = config
        self._entries: Dict[int, MSHREntry] = {}   # address → entry

        # Counters
        self.total_allocations: int = 0
        self.total_merges: int = 0
        self.total_stalls: int = 0          # Times demand couldn't get slot
        self.total_prefetch_evictions: int = 0  # Prefetch evicted for demand
        self.demand_blocked: int = 0        # Demand blocked by prefetch MSHR usage

    @property
    def occupancy(self) -> int:
        return len(self._entries)

    @property
    def utilization(self) -> float:
        if self.config.capacity == 0:
            return 0.0
        return min(1.0, self.occupancy / self.config.capacity)

    @property
    def stall_risk(self) -> MSHRRisk:
        u = self.utilization
        if u < 0.50:
            return MSHRRisk.NONE
        if u < 0.80:
            return MSHRRisk.APPROACHING
        return MSHRRisk.STALLING

    @property
    def is_full(self) -> bool:
        return self.occupancy >= self.config.capacity

    def allocate(
        self,
        address: int,
        is_prefetch: bool,
        cycle: int,
        completion_latency: int = 200,
        workload_id: str = "",
        core_id: int = 0,
    ) -> bool:
        """Try to allocate an MSHR entry.

        Returns True if allocated (or merged), False if dropped/stalled.
        """
        # Align address to cache line
        aligned_addr = (address // 64) * 64

        # Check for secondary miss (merge)
        if self.config.merge_enabled and aligned_addr in self._entries:
            self.total_merges += 1
            return True

        # Check capacity
        if self.is_full:
            if is_prefetch:
                # Prefetch cannot get a slot → dropped
                return False

            # Demand request — try to evict a prefetch entry
            if self.config.demand_priority:
                evicted = self._evict_prefetch()
                if evicted:
                    self.total_prefetch_evictions += 1
                else:
                    # No prefetch to evict — pipeline stalls
                    self.total_stalls += 1
                    self.demand_blocked += 1
                    return False
            else:
                self.total_stalls += 1
                return False

        # Allocate new entry
        entry = MSHREntry(
            address=aligned_addr,
            is_prefetch=is_prefetch,
            cycle_entered=cycle,
            expected_completion=cycle + completion_latency,
            workload_id=workload_id,
            core_id=core_id,
        )
        self._entries[aligned_addr] = entry
        self.total_allocations += 1
        return True

    def complete(self, address: int) -> Optional[MSHREntry]:
        """Complete and remove a specific MSHR entry."""
        aligned_addr = (address // 64) * 64
        return self._entries.pop(aligned_addr, None)

    def tick(self, cycle: int) -> List[MSHREntry]:
        """Complete all entries whose expected_completion <= cycle.

        Returns list of completed entries.
        """
        completed = []
        to_remove = []
        for addr, entry in self._entries.items():
            if entry.expected_completion <= cycle:
                completed.append(entry)
                to_remove.append(addr)
        for addr in to_remove:
            del self._entries[addr]
        return completed

    def has_pending(self, address: int) -> bool:
        """Check if an address already has a pending MSHR entry."""
        aligned_addr = (address // 64) * 64
        return aligned_addr in self._entries

    def _evict_prefetch(self) -> Optional[MSHREntry]:
        """Evict the oldest prefetch entry to make room for demand.

        Returns the evicted entry, or None if no prefetch entries exist.
        """
        oldest_addr: Optional[int] = None
        oldest_cycle: int = float("inf")  # type: ignore[assignment]

        for addr, entry in self._entries.items():
            if entry.is_prefetch and entry.cycle_entered < oldest_cycle:
                oldest_cycle = entry.cycle_entered
                oldest_addr = addr

        if oldest_addr is not None:
            return self._entries.pop(oldest_addr)
        return None

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self._entries.clear()
        self.total_allocations = 0
        self.total_merges = 0
        self.total_stalls = 0
        self.total_prefetch_evictions = 0
        self.demand_blocked = 0
