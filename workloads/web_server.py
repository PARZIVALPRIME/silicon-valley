"""WebServer workload — connection lookup, session, response, TLS, logging.

Archetype: Nginx / Apache request handling
SPEC CPU analog: CloudSuite data-serving

Key characteristic: High temporal locality, small working set.
Aggressive prefetching is UNNECESSARY and potentially harmful
(wastes bandwidth that other workloads need in multi-tenant mode).

Phases:
  1. Connection Lookup: Hash table probe (SEMI-PREDICTABLE, hot keys)
  2. Session Read: Small working set, high reuse (TEMPORAL LOCALITY)
  3. Response Build: Sequential buffer fill (STRIDE-1)
  4. TLS Encrypt: Tight loop over small buffer (VERY HIGH REUSE)
  5. Access Logging: Sequential append-only (STRIDE-1, write)
"""

from __future__ import annotations

import random
from typing import List

from workloads.base import MemoryAccess, WorkloadProfile


# Base addresses
_BASE_CONN = 0x0400_0000
_BASE_SESSION = 0x0500_0000
_BASE_RESPONSE = 0x0600_0000
_BASE_TLS = 0x0700_0000
_BASE_LOG = 0x0800_0000

# Program counters
_PC_CONN = 0x0060_1000
_PC_SESSION = 0x0060_2000
_PC_RESPONSE = 0x0060_3000
_PC_TLS = 0x0060_4000
_PC_LOG = 0x0060_5000

# Connection table
_CONN_TABLE_SLOTS = 4096
_HOT_CONNECTIONS = 32  # 80% of traffic goes to top 32 connections

LINE_SIZE = 64


class WebServer(WorkloadProfile):
    """Web request handling: conn lookup → session → response → TLS → log."""

    workload_id = "web_server"
    description = (
        "Web server request handling: connection hash lookup → session read "
        "→ response buffer build → TLS encryption → access log append"
    )

    def phases(self) -> List[str]:
        return [
            "connection_lookup", "session_read", "response_build",
            "tls_encrypt", "access_logging",
        ]

    def phase_boundaries(self, length: int) -> List[tuple[int, int, str]]:
        c = int(length * 0.20)
        s = int(length * 0.20)
        r = int(length * 0.32)
        t = int(length * 0.16)
        l = length - c - s - r - t
        return [
            (0, c, "connection_lookup"),
            (c, c + s, "session_read"),
            (c + s, c + s + r, "response_build"),
            (c + s + r, c + s + r + t, "tls_encrypt"),
            (c + s + r + t, length, "access_logging"),
        ]

    def generate_trace(self, seed: int, length: int) -> List[MemoryAccess]:
        rng = random.Random(seed)
        trace: List[MemoryAccess] = []
        boundaries = self.phase_boundaries(length)

        for start, end, phase in boundaries:
            count = end - start
            if phase == "connection_lookup":
                trace.extend(self._gen_conn_lookup(rng, count))
            elif phase == "session_read":
                trace.extend(self._gen_session(rng, count))
            elif phase == "response_build":
                trace.extend(self._gen_response(rng, count))
            elif phase == "tls_encrypt":
                trace.extend(self._gen_tls(rng, count))
            elif phase == "access_logging":
                trace.extend(self._gen_logging(rng, count))

        return trace[:length]

    def _gen_conn_lookup(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Connection lookup: hash table with hot/cold distribution.

        80% of lookups hit the top 32 connections → HIGH temporal locality.
        20% hit random slots → unpredictable.
        """
        accesses = []
        for _ in range(count):
            if rng.random() < 0.80:
                # Hot connection — small, predictable set
                slot = rng.randint(0, _HOT_CONNECTIONS - 1)
                predictable = True  # frequently reused
            else:
                # Cold connection — random
                slot = rng.randint(_HOT_CONNECTIONS, _CONN_TABLE_SLOTS - 1)
                predictable = False

            addr = _BASE_CONN + slot * LINE_SIZE
            accesses.append(MemoryAccess(
                pc=_PC_CONN,
                address=addr,
                workload_id=self.workload_id,
                phase="connection_lookup",
                predictable=predictable,
                expected_stride=0,
            ))
        return accesses

    def _gen_session(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Session data read: tiny working set with very high reuse.

        Sessions are ~256 bytes (4 cache lines).  The same session data
        is re-read multiple times per request.  L1 should capture all of it.
        """
        accesses = []
        # Pick a session (from the hot set)
        session_base = _BASE_SESSION + rng.randint(0, 63) * 4 * LINE_SIZE

        for _ in range(count):
            # Access one of 4 cache lines in the session
            line_offset = rng.randint(0, 3) * LINE_SIZE
            addr = session_base + line_offset

            accesses.append(MemoryAccess(
                pc=_PC_SESSION,
                address=addr,
                workload_id=self.workload_id,
                phase="session_read",
                predictable=True,  # high temporal locality
                expected_stride=0,
            ))

            # Occasionally switch sessions (10% chance)
            if rng.random() < 0.10:
                session_base = _BASE_SESSION + rng.randint(0, 63) * 4 * LINE_SIZE

        return accesses

    def _gen_response(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Response buffer: sequential write to build HTTP response."""
        accesses = []
        addr = _BASE_RESPONSE + rng.randint(0, 15) * LINE_SIZE
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_RESPONSE,
                address=addr,
                is_write=True,
                workload_id=self.workload_id,
                phase="response_build",
                predictable=True,
                expected_stride=1,
            ))
            addr += LINE_SIZE
        return accesses

    def _gen_tls(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """TLS encryption: tight loop over small buffer (128B = 2 cache lines).

        Extremely high temporal reuse — the same 2 cache lines are read
        over and over.  Prefetching is pointless here.
        """
        accesses = []
        tls_base = _BASE_TLS + rng.randint(0, 31) * 2 * LINE_SIZE
        for _ in range(count):
            offset = rng.randint(0, 1) * LINE_SIZE
            accesses.append(MemoryAccess(
                pc=_PC_TLS,
                address=tls_base + offset,
                workload_id=self.workload_id,
                phase="tls_encrypt",
                predictable=True,  # temporal, not spatial
                expected_stride=0,
            ))
        return accesses

    def _gen_logging(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Access log: append-only sequential write."""
        accesses = []
        addr = _BASE_LOG
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_LOG,
                address=addr,
                is_write=True,
                workload_id=self.workload_id,
                phase="access_logging",
                predictable=True,
                expected_stride=1,
            ))
            addr += LINE_SIZE
        return accesses
