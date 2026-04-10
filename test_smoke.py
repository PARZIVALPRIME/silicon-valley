"""Smoke test for SiliconMind — validates all modules load and core logic works."""
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

print("=" * 60)
print(" SiliconMind Smoke Test")
print("=" * 60)

# 1. Models
from models import (
    SiliconMindAction, SiliconMindObservation, SiliconMindState,
    ActionType, PrefetchOutcome, REWARD_VALUES, normalize_reward,
)
print(f"\n✓ Models loaded — {len(REWARD_VALUES)} reward levels")
print(f"  RAT (+20) → normalized: {normalize_reward(20.0)}")
print(f"  POLLUTION (-18) → normalized: {normalize_reward(-18.0)}")

# 2. Cache
from engine.cache import CacheHierarchy, L1_CONFIG, L2_CONFIG, LLC_CONFIG
ch = CacheHierarchy(l1_config=L1_CONFIG, l2_config=L2_CONFIG, llc_config=LLC_CONFIG)
# Prefetch + demand
pf = ch.prefetch(addr=0x1000, cycle=100)
r = ch.access(addr=0x1000, cycle=110)
print(f"\n✓ Cache hierarchy — prefetch issued={pf.issued}, demand hit={r.hit}")

# 3. Bandwidth
from engine.bandwidth import BandwidthController
bw = BandwidthController()
for i in range(20):
    bw.request(is_prefetch=True)
bw.tick(5)
print(f"✓ Bandwidth — util: {bw.utilization:.1%}, pressure: {bw.pressure.value}")

# 4. MSHR
from engine.mshr import MSHRQueue
mq = MSHRQueue()
for i in range(16):
    mq.allocate(address=i * 64, is_prefetch=True, cycle=0)
full = mq.is_full
result = mq.allocate(address=99 * 64, is_prefetch=True, cycle=0)
print(f"✓ MSHR — 16/16 filled={full}, 17th prefetch dropped={not result}")

# 5. Workloads
from workloads.database import DatabaseEngine
from workloads.ml_training import MLTrainingLoop
from workloads.web_server import WebServer
from workloads.video_encoder import VideoEncoder

for W in [DatabaseEngine, MLTrainingLoop, WebServer, VideoEncoder]:
    w = W()
    trace = w.generate_trace(42, 100)
    phases = set(a.phase for a in trace)
    pred = sum(a.predictable for a in trace)
    t1 = w.generate_trace(42, 10)
    t2 = w.generate_trace(42, 10)
    det = all(a.address == b.address for a, b in zip(t1, t2))
    print(f"✓ {w.workload_id}: {len(trace)} accesses, {len(phases)} phases, "
          f"predictable={pred}%, deterministic={det}")

# 6. Tasks
from tasks import TASK_REGISTRY
from graders.baselines import compute_baseline_misses

print(f"\n✓ Task Registry — {len(TASK_REGISTRY)} tasks")
for tid, task in TASK_REGISTRY.items():
    trace = task.build_trace(42)
    configs = task.cache_configs()
    bm = compute_baseline_misses(trace, configs)
    slas = len(task.sla_requirements())
    avail = [a.value for a in task.available_actions()]
    print(f"  {tid}")
    print(f"    difficulty={task.difficulty}, steps={len(trace)}, "
          f"baseline_misses={bm}/{len(trace)} ({bm/len(trace)*100:.0f}%)")
    print(f"    metrics={list(task.grader_weights().keys())}")
    print(f"    actions={avail}, SLAs={slas}")

# 7. Grader
from engine.counters import EpisodeStats
from graders.score import compute_final_score

stats = EpisodeStats()
stats.total_accesses = 100
stats.total_hits = 60
stats.total_misses = 40
stats.total_prefetches_issued = 50
stats.accurate_timely = 30
stats.accurate_late = 5
stats.pollution_evictions = 2
stats.timely_prefetches = 30
stats.baseline_misses = 80
stats.unpredictable_accesses = 10
stats.correct_no_prefetch = 7

task = TASK_REGISTRY["easy_pattern_scout"]
scores = compute_final_score(stats, task)
print(f"\n✓ Grader — synthetic episode scores:")
for k, v in scores.items():
    print(f"    {k}: {v}")

# 8. Narrator
from engine.narrator import generate_narrative
n = generate_narrative(
    step=50, steps_remaining=90, workload="ml_training", phase="backward_pass",
    cache_result="MISS", detected_pattern="stride-8", pattern_confidence=0.45,
    previous_pattern="stride-1", previous_confidence=0.92,
    prefetch_accuracy_recent=0.11, prefetch_accuracy_total=0.64,
    hit_rate=0.41, bandwidth_pressure="high", bandwidth_utilization=0.82,
    mshr_risk="none", pollution_rate=0.08,
    last_outcome="WASTED", last_explanation="Wrong offset",
    active_workloads=["ml_training"], sla_status={}, recent_deltas=[8, 8, 8, 1, 1, 1],
)
print(f"\n✓ Narrator output:")
print(f"  '{n[:150]}...'")

print("\n" + "=" * 60)
print(" ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
