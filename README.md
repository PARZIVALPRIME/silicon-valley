# SiliconMind — Memory Subsystem Intelligence Gym 🧠⚡

> *The first OpenEnv environment that benchmarks whether AI agents can optimize CPU cache prefetching — a $100B+ industry problem affecting every modern processor.*

## What Is This?

SiliconMind simulates a realistic CPU memory subsystem where an AI agent acts as a **hardware prefetcher**.  At each step, the agent observes a memory access (address, pattern, cache state, bandwidth pressure) and decides:

- **Should I prefetch?** If so, which cache lines?
- **How aggressively?** (1-4 prefetches per access)
- **How to partition shared resources?** (LLC ways, bandwidth allocation)

The environment grades the agent on **industry-standard metrics** used by Intel and AMD to evaluate real hardware prefetchers.

## Why This Matters

Modern CPUs waste **30-50% of cycles** waiting for data from memory (the "Memory Wall").  Cache prefetching is the primary weapon against this bottleneck, but designing good prefetchers is an unsolved problem worth $50B+/year in wasted compute.

SiliconMind tests whether frontier LLMs can reason about memory access patterns well enough to outperform simple heuristics.

## Key Innovations

| Innovation | Description |
|---|---|
| 🧠 **LLM-Native Observations** | Natural language narrative + structured data designed for LLM reasoning |
| 🎯 **Pythia 9-Level Reward** | Research-grade reward classification from MICRO '21 (RAT/RAL/RIN/POLLUTION) |
| 🔬 **Prefetch Forensics** | Complete lifecycle tracking: ISSUED → QUEUED → FILLED → USEFUL/WASTED |
| ⚖️ **Counterfactual Baselines** | Automatic no-prefetch baseline for coverage scoring |
| 📡 **Regime Detection** | Pattern confidence drops on workload transitions |
| 🏛️ **Jain's Fairness Index** | Industry-standard multi-tenant fairness metric |
| 🔄 **Cross-Core Pollution** | Cross-core LLC eviction tracking with 2× penalty |

## Tasks

| Task | Difficulty | Description | Expected Score |
|---|---|---|---|
| Pattern Scout | 🟢 Easy | Stride patterns + noise. L1 only. | 0.40-0.60 |
| Workload Whisperer | 🟡 Medium | ML training with phase changes. L1+L2. | 0.25-0.40 |
| Noisy Neighbor | 🔴 Hard | 3 workloads competing for shared LLC + bandwidth. | 0.10-0.25 |
| Silicon Architect | ⚫ Extreme | Multi-core, regime change, 8 MSHRs, cross-core pollution. | 0.00-0.12 |

## Action Space

```json
{
  "type": "prefetch | no_prefetch | set_aggressiveness | throttle_bandwidth | partition_cache",
  "target": "1,2 | skip | 1-4 | 0.0-1.0 | web:6,db:6,video:4",
  "detail": "optional reasoning"
}
```

## Observation Space

Each step returns:
- **`narrative`**: 2-3 sentence natural language summary optimized for LLM reasoning
- **`current_access`**: PC, address, cache result, workload, phase
- **`pattern_analysis`**: Detected pattern, confidence, phase change signals
- **`performance`**: Accuracy (recent + total), hit rate, reward breakdown
- **`system_state`**: Bandwidth pressure, MSHR utilization, pollution rate
- **`sla_status`**: Per-workload SLA compliance (multi-tenant tasks)

## Reward System

Based on Pythia (MICRO '21) — 9-level classification:

| Level | Condition | Raw Reward |
|---|---|---|
| RAT | Accurate + Timely | +20 |
| RAL | Accurate + Late | +12 |
| RESTRAINT | Correct NO_PREFETCH | +8 |
| REDUNDANT | Already in cache | -2 |
| RNP_H | No-prefetch, high BW | -2 |
| RNP_L | No-prefetch, low BW | -4 |
| RIN_L | Inaccurate, low BW | -8 |
| RIN_H | Inaccurate, high BW | -14 |
| POLLUTION | Evicted useful line | -18 |

Normalized to [0, 1] for OpenEnv compliance.

## Quick Start

```bash
# Install
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export API_BASE_URL="https://api.example.com/v1"
export API_KEY="your-key"
python inference.py
```

## Docker

```bash
docker build -t siliconmind .
docker run -p 7860:7860 siliconmind
```

## Hardware Configuration

Matches CRL-Pythia (MICRO '21) paper:

| Component | Size | Associativity | Latency |
|---|---|---|---|
| L1 Data Cache | 32 KB | 8-way | 4 cycles |
| L2 Cache | 256 KB | 8-way | 12 cycles |
| LLC (shared) | 2 MB | 16-way | 40 cycles |
| DRAM | — | — | 200 cycles |

## Grading

All grading is **pure deterministic math** — no LLM-based evaluation:
- Accuracy = useful_prefetches / total_prefetches
- Coverage = (baseline_misses - agent_misses) / baseline_misses
- Fairness = Jain's Index: (Σxᵢ)² / (n × Σxᵢ²)
- All scores clamped to (0.01, 0.99)

## References

- Bera et al., "Pythia: A Customizable Hardware Prefetching Framework Using Online Reinforcement Learning," MICRO 2021
- CRL-Pythia: Coordinated Reinforcement Learning Prefetching Architecture for Multicore Systems (arXiv:2509.10719)

## License

MIT
