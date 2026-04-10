"""Test the SiliconMind REST API — reset/step/state endpoints.

Handles the OpenEnv response format:
  {observation: {...}, reward: float|null, done: bool}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx

BASE = "http://localhost:7860"


def parse_response(r):
    """Parse OpenEnv response — handles both wrapped and flat formats."""
    data = r.json()
    if "observation" in data and isinstance(data["observation"], dict):
        # Wrapped format: {observation: {...}, reward: float, done: bool}
        obs = data["observation"]
        obs["_reward_float"] = data.get("reward")
        obs["_done_top"] = data.get("done", False)
        return obs
    return data


def test_root():
    print("=== ROOT ===")
    r = httpx.get(f"{BASE}/")
    assert r.status_code == 200, f"Root failed: {r.status_code}"
    data = r.json()
    print(f"  Name: {data['name']}")
    print(f"  Tasks: {data['tasks']}")
    assert len(data["tasks"]) == 4
    print("  PASS\n")


def test_health():
    print("=== HEALTH ===")
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    print(f"  Status: {r.json()}")
    print("  PASS\n")


def test_reset_step_state():
    print("=== RESET (easy_pattern_scout) ===")
    r = httpx.post(f"{BASE}/reset", json={"task_id": "easy_pattern_scout", "seed": 42}, timeout=30)
    assert r.status_code == 200, f"Reset failed: {r.status_code} {r.text[:500]}"
    obs = parse_response(r)
    print(f"  Step: {obs['step_number']}, Remaining: {obs['steps_remaining']}")
    print(f"  Task: {obs['task_id']}")
    print(f"  Workload: {obs['current_workload']}, Phase: {obs['current_phase']}")
    print(f"  Narrative: {obs['narrative'][:150]}...")
    print(f"  Available: {obs['available_actions']}")
    print(f"  Done: {obs.get('done', obs.get('_done_top'))}")
    print("  PASS\n")

    # Step with prefetch
    print("=== STEP (prefetch +1) ===")
    r2 = httpx.post(f"{BASE}/step", json={"action": {"type": "prefetch", "target": "1"}}, timeout=30)
    assert r2.status_code == 200, f"Step failed: {r2.status_code} {r2.text[:500]}"
    obs2 = parse_response(r2)
    print(f"  Step: {obs2['step_number']}, Remaining: {obs2['steps_remaining']}")
    rw = obs2.get("_reward_float", obs2.get("reward", 0))
    print(f"  Reward (float): {rw}")
    rd = obs2.get("reward_detail", {})
    print(f"  Outcome: {rd.get('outcome', 'N/A')}")
    print(f"  Explanation: {rd.get('explanation', 'N/A')[:100]}")
    print(f"  Hit rate: {obs2['cache_stats']['hit_rate']}")
    print(f"  Narrative: {obs2['narrative'][:150]}...")
    if rw is not None:
        assert 0.0 <= float(rw) <= 1.0, f"Reward {rw} out of range!"
    print("  PASS\n")

    # Step with no_prefetch
    print("=== STEP (no_prefetch) ===")
    r3 = httpx.post(f"{BASE}/step", json={"action": {"type": "no_prefetch", "target": "skip"}}, timeout=30)
    assert r3.status_code == 200, f"Step failed: {r3.status_code} {r3.text[:500]}"
    obs3 = parse_response(r3)
    rw3 = obs3.get("_reward_float", obs3.get("reward", 0))
    rd3 = obs3.get("reward_detail", {})
    print(f"  Step: {obs3['step_number']}, Reward: {rw3}, Outcome: {rd3.get('outcome', 'N/A')}")
    print("  PASS\n")

    # State
    print("=== STATE ===")
    r4 = httpx.get(f"{BASE}/state", timeout=30)
    assert r4.status_code == 200, f"State failed: {r4.status_code}"
    state = r4.json()
    print(f"  State: {state}")
    print("  PASS\n")


def test_full_episode():
    """Run a complete easy episode to verify grading."""
    print("=== FULL EPISODE (easy_pattern_scout) ===")
    r = httpx.post(f"{BASE}/reset", json={"task_id": "easy_pattern_scout", "seed": 42}, timeout=30)
    obs = parse_response(r)
    steps = 0
    total_reward = 0.0

    while steps < 200:
        done = obs.get("done", obs.get("_done_top", False))
        if done:
            break

        conf = obs["pattern_analysis"]["confidence"]
        pattern = obs["pattern_analysis"]["detected_pattern"]

        if conf > 0.5 and "stride" in pattern:
            action = {"type": "prefetch", "target": "1"}
        else:
            action = {"type": "no_prefetch", "target": "skip"}

        r = httpx.post(f"{BASE}/step", json={"action": action}, timeout=30)
        obs = parse_response(r)
        rw = obs.get("_reward_float", obs.get("reward", 0)) or 0
        total_reward += float(rw)
        steps += 1

    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Done: {obs.get('done', obs.get('_done_top'))}")

    info = obs.get("info", {})
    if "final_scores" in info:
        scores = info["final_scores"]
        print(f"  Final scores:")
        for k, v in scores.items():
            print(f"    {k}: {v}")
        for k, v in scores.items():
            assert 0.01 <= v <= 0.99, f"Score {k}={v} out of range!"
        print(f"  All scores in (0.01, 0.99) range: PASS")
    else:
        print("  WARNING: No final_scores in info")

    if "episode_stats" in info:
        es = info["episode_stats"]
        print(f"  Episode stats:")
        for k, v in es.items():
            print(f"    {k}: {v}")

    print("  PASS\n")


def test_all_tasks_reset():
    """Test that all 4 tasks reset correctly."""
    print("=== ALL TASKS RESET ===")
    tasks = [
        "easy_pattern_scout",
        "medium_workload_whisperer",
        "hard_noisy_neighbor",
        "extreme_silicon_architect",
    ]
    for tid in tasks:
        r = httpx.post(f"{BASE}/reset", json={"task_id": tid, "seed": 42}, timeout=30)
        assert r.status_code == 200, f"Reset {tid} failed: {r.status_code} {r.text[:300]}"
        obs = parse_response(r)
        print(f"  {tid}: steps_remaining={obs['steps_remaining']}, "
              f"actions={obs['available_actions']}")
    print("  ALL PASS\n")


def test_medium_full_episode():
    """Run medium episode with phase adaptation."""
    print("=== FULL EPISODE (medium_workload_whisperer) ===")
    r = httpx.post(f"{BASE}/reset", json={"task_id": "medium_workload_whisperer", "seed": 42}, timeout=30)
    obs = parse_response(r)
    steps = 0

    while steps < 600:
        done = obs.get("done", obs.get("_done_top", False))
        if done:
            break

        conf = obs["pattern_analysis"]["confidence"]
        pattern = obs["pattern_analysis"]["detected_pattern"]
        bw = obs["bandwidth"]["pressure"]

        if conf > 0.5 and "stride" in pattern and bw not in ("high", "critical"):
            action = {"type": "prefetch", "target": "1"}
        elif bw in ("high", "critical"):
            action = {"type": "set_aggressiveness", "target": "1"}
        else:
            action = {"type": "no_prefetch", "target": "skip"}

        r = httpx.post(f"{BASE}/step", json={"action": action}, timeout=30)
        obs = parse_response(r)
        steps += 1

    print(f"  Steps: {steps}")
    info = obs.get("info", {})
    if "final_scores" in info:
        scores = info["final_scores"]
        print(f"  Final scores:")
        for k, v in scores.items():
            print(f"    {k}: {v}")
        for k, v in scores.items():
            assert 0.01 <= v <= 0.99, f"Score {k}={v} out of range!"
    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" SiliconMind API Integration Tests")
    print("=" * 60)
    print()
    test_root()
    test_health()
    test_reset_step_state()
    test_all_tasks_reset()
    test_full_episode()
    test_medium_full_episode()
    print("=" * 60)
    print(" ALL API TESTS PASSED!")
    print("=" * 60)
