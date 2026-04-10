"""MLTrainingLoop workload — forward/backward/gradient/optimizer phases.

Archetype: PyTorch DNN training iteration (weight_dim=64, batch_size=32)
SPEC CPU analog: 638.imagick_s (matrix-heavy), 654.roms_s (scientific compute)

The killer challenge: forward pass is stride-1 (row-major), backward pass
is stride-64 (column-major on the SAME matrix).  The agent must detect
this pattern inversion and adapt.

Phases:
  1. Forward Pass: Row-major matrix access (stride-1, VERY PREDICTABLE)
  2. Activation Store: Sequential streaming writes (PREDICTABLE but useless for cache)
  3. Backward Pass: Column-major access (stride-64, CACHE-HOSTILE)
  4. Gradient Update: Strided-8 scattered updates (SEMI-PREDICTABLE)
  5. Optimizer Step: 3 interleaved arrays—weights/momentum/variance (COMPLEX)
"""

from __future__ import annotations

import random
from typing import List

from workloads.base import MemoryAccess, WorkloadProfile


# Base addresses (16 MB regions)
_BASE_WEIGHTS = 0x0100_0000     # Weight matrix
_BASE_ACTIVATIONS = 0x0200_0000
_BASE_GRADIENTS = 0x0300_0000
_BASE_MOMENTUM = 0x0310_0000
_BASE_VARIANCE = 0x0320_0000

# Matrix dimensions
_WEIGHT_DIM = 64   # 64×64 matrix
_BATCH_SIZE = 32

# Program counters
_PC_FORWARD = 0x0050_1000
_PC_ACTIVATION = 0x0050_2000
_PC_BACKWARD = 0x0050_3000
_PC_GRADIENT = 0x0050_4000
_PC_OPTIMIZER = 0x0050_5000

LINE_SIZE = 64


class MLTrainingLoop(WorkloadProfile):
    """DNN training: forward → activation → backward → gradient → optimizer."""

    workload_id = "ml_training"
    description = (
        "ML training loop: row-major forward pass → activation store → "
        "column-major backward pass → strided gradient update → "
        "3-array Adam optimizer step"
    )

    def phases(self) -> List[str]:
        return [
            "forward_pass", "activation_store", "backward_pass",
            "gradient_update", "optimizer_step",
        ]

    def phase_boundaries(self, length: int) -> List[tuple[int, int, str]]:
        """Forward: 20%, Activation: 8%, Backward: 32%, Gradient: 20%, Optimizer: 20%"""
        f = int(length * 0.20)
        a = int(length * 0.08)
        b = int(length * 0.32)
        g = int(length * 0.20)
        o = length - f - a - b - g
        return [
            (0, f, "forward_pass"),
            (f, f + a, "activation_store"),
            (f + a, f + a + b, "backward_pass"),
            (f + a + b, f + a + b + g, "gradient_update"),
            (f + a + b + g, length, "optimizer_step"),
        ]

    def generate_trace(self, seed: int, length: int) -> List[MemoryAccess]:
        rng = random.Random(seed)
        trace: List[MemoryAccess] = []
        boundaries = self.phase_boundaries(length)

        for start, end, phase in boundaries:
            count = end - start
            if phase == "forward_pass":
                trace.extend(self._gen_forward(rng, count))
            elif phase == "activation_store":
                trace.extend(self._gen_activation(rng, count))
            elif phase == "backward_pass":
                trace.extend(self._gen_backward(rng, count))
            elif phase == "gradient_update":
                trace.extend(self._gen_gradient(rng, count))
            elif phase == "optimizer_step":
                trace.extend(self._gen_optimizer(rng, count))

        return trace[:length]

    def _gen_forward(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Forward pass: row-major matrix traversal.

        Access pattern: row 0 left-to-right, then row 1, etc.
        Stride-1 in cache lines (64B), wrapping at row boundaries.
        HIGHLY PREDICTABLE — the easy win for a prefetcher.
        """
        accesses = []
        addr = _BASE_WEIGHTS + rng.randint(0, 3) * LINE_SIZE
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_FORWARD,
                address=addr,
                workload_id=self.workload_id,
                phase="forward_pass",
                predictable=True,
                expected_stride=1,
            ))
            addr += LINE_SIZE  # stride-1

        return accesses

    def _gen_activation(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Activation store: sequential streaming writes.

        Pattern: stride-1 sequential, but write-only.
        Predictable stride, but writes don't benefit from prefetch (write-allocate).
        """
        accesses = []
        addr = _BASE_ACTIVATIONS
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_ACTIVATION,
                address=addr,
                is_write=True,
                workload_id=self.workload_id,
                phase="activation_store",
                predictable=True,
                expected_stride=1,
            ))
            addr += LINE_SIZE
        return accesses

    def _gen_backward(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Backward pass: COLUMN-MAJOR matrix traversal.

        Accesses the SAME weight matrix as forward pass, but in
        column-major order: stride = row_width × element_size.

        For a 64×64 matrix with 8-byte elements:
          stride = 64 rows × (64 elements × 8 bytes / 64 bytes/line) = 64 cache lines

        This is CACHE-HOSTILE: each access skips 4KB (64 × 64 bytes),
        far exceeding the L1 cache capacity.

        The agent's previous stride-1 strategy will FAIL here.
        Its accuracy will crash, and it must adapt or switch to NO_PREFETCH.
        """
        accesses = []
        # stride = row_width in cache lines
        # 64 elements per row × 8 bytes = 512 bytes = 8 cache lines per row
        stride_lines = _WEIGHT_DIM * 8 // LINE_SIZE  # = 8 cache lines
        addr = _BASE_WEIGHTS
        col = 0

        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_BACKWARD,
                address=addr,
                workload_id=self.workload_id,
                phase="backward_pass",
                predictable=True,       # Predictable IF you know the stride
                expected_stride=stride_lines,
            ))
            addr += stride_lines * LINE_SIZE  # column-major stride

            # Wrap around when we've gone through all rows
            if addr > _BASE_WEIGHTS + _WEIGHT_DIM * _WEIGHT_DIM * 8:
                col += 1
                addr = _BASE_WEIGHTS + col * 8

        return accesses

    def _gen_gradient(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Gradient accumulation: strided-8 updates across weight tensor.

        Pattern: update every 8th cache line (skip connections).
        Semi-predictable stride-8 pattern.
        """
        accesses = []
        addr = _BASE_GRADIENTS
        stride = 8  # stride-8 in cache lines
        for _ in range(count):
            accesses.append(MemoryAccess(
                pc=_PC_GRADIENT,
                address=addr,
                workload_id=self.workload_id,
                phase="gradient_update",
                predictable=True,
                expected_stride=stride,
            ))
            addr += stride * LINE_SIZE
            # Wrap around
            if addr > _BASE_GRADIENTS + 0x10_0000:
                addr = _BASE_GRADIENTS + rng.randint(0, 7) * LINE_SIZE
        return accesses

    def _gen_optimizer(self, rng: random.Random, count: int) -> List[MemoryAccess]:
        """Adam optimizer: 3 interleaved array reads.

        Pattern: read weights[i], momentum[i], variance[i] — repeat.
        Three interleaved stride-1 streams.
        Agent must learn the 3-way interleave pattern.
        """
        accesses = []
        w_addr = _BASE_WEIGHTS
        m_addr = _BASE_MOMENTUM
        v_addr = _BASE_VARIANCE
        arrays = [
            (_BASE_WEIGHTS, _PC_OPTIMIZER),
            (_BASE_MOMENTUM, _PC_OPTIMIZER),
            (_BASE_VARIANCE, _PC_OPTIMIZER),
        ]
        idx = 0

        for i in range(count):
            arr_idx = i % 3  # round-robin across 3 arrays
            base, pc = arrays[arr_idx]
            offset = (i // 3) * LINE_SIZE
            addr = base + offset

            accesses.append(MemoryAccess(
                pc=pc,
                address=addr,
                workload_id=self.workload_id,
                phase="optimizer_step",
                predictable=True,   # Predictable IF you know the interleave
                expected_stride=1 if arr_idx == 0 else 0,  # complex pattern
            ))

        return accesses
