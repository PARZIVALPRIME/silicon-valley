"""Base classes for workload trace generation.

Every workload produces a deterministic sequence of MemoryAccess objects
given the same seed.  Each access includes:
  - Program counter (PC): identifies the instruction source
  - Memory address: the data address being accessed
  - Phase: which algorithmic phase generated this access
  - Predictability: whether this access is part of a predictable pattern

The phase and predictability fields are used by the grader to evaluate
the agent's restraint (correct NO_PREFETCH on unpredictable accesses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


# ───────────────────────────────────────────────────────────────────────────
# Memory Access — the fundamental unit of a trace
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryAccess:
    """A single memory access event in a workload trace."""

    pc: int                           # Program counter (instruction address)
    address: int                      # Data memory address (cache-line aligned)
    is_write: bool = False
    workload_id: str = "default"
    phase: str = "unknown"            # e.g., "btree_descent", "forward_pass"
    predictable: bool = True          # False for random/hash patterns
    expected_stride: int = 0          # Ground-truth stride (for grader)


# ───────────────────────────────────────────────────────────────────────────
# Workload Profile — abstract base for all workloads
# ───────────────────────────────────────────────────────────────────────────

class WorkloadProfile(ABC):
    """Abstract base for workload trace generators.

    Each subclass defines a specific application archetype
    (database, ML training, web server, etc.) and generates
    a deterministic trace of MemoryAccess objects.
    """

    workload_id: str = "default"
    description: str = ""

    @abstractmethod
    def generate_trace(self, seed: int, length: int) -> List[MemoryAccess]:
        """Generate a deterministic trace of `length` memory accesses.

        Same seed + same length → identical trace (byte-for-byte).
        """
        ...

    @abstractmethod
    def phases(self) -> List[str]:
        """Return the list of phases in this workload."""
        ...

    @abstractmethod
    def phase_boundaries(self, length: int) -> List[tuple[int, int, str]]:
        """Return (start_step, end_step, phase_name) for each phase.

        This is used by the narrator to detect phase transitions.
        """
        ...
