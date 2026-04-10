"""VideoEncoder workload — reference frame, motion estimation, DCT, entropy, deblock.

Archetype: H.264 / H.265 encoding pipeline
SPEC CPU analog: 625.x264_s (video compression)

Key characteristic: Reference frame loading can SATURATE memory bandwidth.
In multi-tenant mode, aggressive video prefetching destroys performance
for all other workloads.

Phases:
  1. Reference Frame Load: Large sequential read (4KB, 64 cache lines)
  2. Motion Estimation: Diamond search pattern (semi-random ±16 lines)
  3. DCT Transform: 8×8 block strided access
  4. Entropy Coding: Sequential with variable length
  5. Deblock Filter: 2D stencil across macroblock boundaries
"""

from __future__ import annotations

import random
from typing import List

from workloads.base import MemoryAccess, WorkloadProfile


# Base addresses
_BASE_FRAME = 0x0900_0000      # Reference frame (large, sequential)
_BASE_SEARCH = 0x0A00_0000     # Search window
_BASE_DCT = 0x0B00_0000        # DCT coefficients
_BASE_ENTROPY = 0x0B10_0000
_BASE_DEBLOCK = 0x0B20_0000

# Program counters
_PC_FRAME = 0x0070_1000
_PC_MOTION = 0x0070_2000
_PC_DCT = 0x0070_3000
_PC_ENTROPY = 0x0070_4000
_PC_DEBLOCK = 0x0070_5000

# Frame dimensions
_FRAME_WIDTH_LINES = 64   # cache lines per row
_MACROBLOCK_SIZE = 16     # pixels (2 cache lines at 8 bytes/pixel)

LINE_SIZE = 64


class VideoEncoder(WorkloadProfile):
    """Video encoding: frame → motion est. → DCT → entropy → deblock."""

    workload_id = "video_encoder"
    description = (
        "Video encoding pipeline: reference frame load → diamond motion "
        "estimation → 8×8 DCT transform → entropy coding → deblock filter"
    )

    def phases(self) -> List[str]:
        return [
            "reference_frame", "motion_estimation", "dct_transform",
            "entropy_coding", "deblock_filter",
        ]

    def phase_boundaries(self, length: int) -> List[tuple[int, int, str]]:
        f = int(length * 0.32)   # 32% frame (bandwidth-heavy)
        m = int(length * 0.32)   # 32% motion (semi-random)
        d = int(length * 0.16)   # 16% DCT
        e = int(length * 0.12)   # 12% entropy
        b = length - f - m - d - e
        return [
            (0, f, "reference_frame"),
            (f, f + m, "motion_estimation"),
            (f + m, f + m + d, "dct_transform"),
            (f + m + d, f + m + d + e, "entropy_coding"),
            (f + m + d + e, length, "deblock_filter"),
        ]

    def generate_trace(self, seed: int, length: int) -> List[MemoryAccess]:
        rng = random.Random(seed)
        trace: List[MemoryAccess] = []
        boundaries = self.phase_boundaries(length)

        for start, end, phase in boundaries:
            count = end - start
            if phase == "reference_frame":
                trace.extend(self._gen_frame(rng, count))
            elif phase == "motion_estimation":
                trace.extend(self._gen_motion(rng, count))
            elif phase == "dct_transform":
                trace.extend(self._gen_dct(rng, count))
            elif phase == "entropy_coding":
                trace.extend(self._gen_entropy(rng, count))
            elif phase == "deblock_filter":
                trace.extend(self._gen_deblock(rng, count))

        return trace[:length]

    def _gen_frame(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Reference frame load: large sequential read.

        Reads through an entire frame buffer sequentially.
        Stride-1 pattern, BUT requires enormous bandwidth
        (64 cache lines = 4KB per macroblock row).

        In multi-tenant mode, this alone can saturate the memory bus.
        """
        accesses = []
        addr = _BASE_FRAME
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_FRAME,
                address=addr,
                workload_id=self.workload_id,
                phase="reference_frame",
                predictable=True,
                expected_stride=1,
            ))
            addr += LINE_SIZE
        return accesses

    def _gen_motion(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Motion estimation: diamond search pattern.

        For each macroblock, search neighboring positions in a diamond shape:
          - Start at (x, y)
          - Check (x±1, y), (x, y±1), then (x±2, y), (x, y±2), etc.

        The jumps are semi-regular but not perfectly predictable.
        Row stride varies: sometimes +1 line, sometimes +64.
        """
        accesses = []
        # Start at a random macroblock position
        mb_x = rng.randint(0, 30)
        mb_y = rng.randint(0, 30)

        diamond_offsets = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
        ]

        for i in range(count):
            # Pick a diamond offset
            off_idx = i % len(diamond_offsets)
            dx, dy = diamond_offsets[off_idx]
            x = max(0, mb_x + dx)
            y = max(0, mb_y + dy)

            addr = _BASE_SEARCH + (y * _FRAME_WIDTH_LINES + x) * LINE_SIZE

            accesses.append(MemoryAccess(
                pc=_PC_MOTION,
                address=addr,
                workload_id=self.workload_id,
                phase="motion_estimation",
                predictable=False,  # Diamond pattern is semi-random
                expected_stride=0,
            ))

            # Move to next macroblock every ~13 accesses (one diamond)
            if off_idx == len(diamond_offsets) - 1:
                mb_x = (mb_x + 1) % 32
                if mb_x == 0:
                    mb_y = (mb_y + 1) % 32

        return accesses

    def _gen_dct(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """DCT transform: 8×8 block access with stride-8.

        Each 8×8 block is processed row by row, then the next block.
        Stride within block: 1 cache line per row (stride-1 for 8 accesses)
        Stride between blocks: varies
        """
        accesses = []
        block_idx = 0
        in_block = 0
        block_base = _BASE_DCT

        for _ in range(count):
            addr = block_base + in_block * LINE_SIZE
            accesses.append(MemoryAccess(
                pc=_PC_DCT,
                address=addr,
                workload_id=self.workload_id,
                phase="dct_transform",
                predictable=True,
                expected_stride=1 if in_block > 0 else 8,
            ))
            in_block += 1
            if in_block >= 8:  # finished one 8×8 block
                in_block = 0
                block_idx += 1
                block_base = _BASE_DCT + block_idx * 8 * LINE_SIZE

        return accesses

    def _gen_entropy(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Entropy coding: sequential with small stride."""
        accesses = []
        addr = _BASE_ENTROPY
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_ENTROPY,
                address=addr,
                workload_id=self.workload_id,
                phase="entropy_coding",
                predictable=True,
                expected_stride=1,
            ))
            addr += LINE_SIZE
        return accesses

    def _gen_deblock(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Deblock filter: 2D stencil across macroblock boundaries.

        Accesses pixels at the edge of two adjacent macroblocks:
          left_mb[right_edge], right_mb[left_edge]
        Pattern: alternates between stride-1 and stride-64.
        """
        accesses = []
        mb_row = 0
        mb_col = 0
        reading_left = True

        for _ in range(count):
            if reading_left:
                # Right edge of left macroblock
                addr = _BASE_DEBLOCK + (mb_row * _FRAME_WIDTH_LINES + mb_col * 2 + 1) * LINE_SIZE
                stride = 1
            else:
                # Left edge of right macroblock
                addr = _BASE_DEBLOCK + (mb_row * _FRAME_WIDTH_LINES + (mb_col + 1) * 2) * LINE_SIZE
                stride = _FRAME_WIDTH_LINES  # jump to next row

            accesses.append(MemoryAccess(
                pc=_PC_DEBLOCK,
                address=addr,
                workload_id=self.workload_id,
                phase="deblock_filter",
                predictable=True,
                expected_stride=stride,
            ))

            reading_left = not reading_left
            if not reading_left:
                mb_col += 1
                if mb_col >= 31:
                    mb_col = 0
                    mb_row += 1

        return accesses
