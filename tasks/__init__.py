"""Task registry — maps task_id strings to task instances.

Used by the environment to look up task configuration during reset().
"""

from __future__ import annotations

from typing import Dict

from tasks.base import BaseTask
from tasks.task_scout import PatternScoutTask
from tasks.task_whisperer import WorkloadWhispererTask
from tasks.task_arbitrator import NoisyNeighborTask
from tasks.task_architect import SiliconArchitectTask


TASK_REGISTRY: Dict[str, BaseTask] = {
    "easy_pattern_scout": PatternScoutTask(),
    "medium_workload_whisperer": WorkloadWhispererTask(),
    "hard_noisy_neighbor": NoisyNeighborTask(),
    "extreme_silicon_architect": SiliconArchitectTask(),
}

DEFAULT_TASK = "easy_pattern_scout"
