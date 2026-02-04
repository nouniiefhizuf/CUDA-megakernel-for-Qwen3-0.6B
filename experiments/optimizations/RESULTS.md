# Megakernel Optimization Experiments

## Goal
Reduce grid.sync() calls from 8 per layer to theoretical minimum of 5 per layer.

Current: 8 syncs/layer x 28 layers = 224 syncs per decode step.

## Optimization 1: Redundant RMSNorm

**Status**: Implemented and tested

**Approach**: Have ALL 82 blocks compute RMSNorm redundantly instead of only block 0. The hidden state is only 1024 elements (2KB) - fits in L2 cache after first block reads it.

**Syncs eliminated**: 56 (2 per layer x 28 layers)

**Results**:
```
Position    Original    Optimized    Speedup
-------------------------------------------------
1           5.655ms     3.982ms      1.42x
10          5.708ms     4.006ms      1.42x
50          5.819ms     4.107ms      1.42x
100         5.936ms     4.227ms      1.40x
200         6.202ms     6.883ms      0.90x
-------------------------------------------------
Average     5.864ms     4.641ms      1.26x

Original:  170.5 tok/s
Optimized: 215.5 tok/s
Improvement: +44.9 tok/s (26.3%)
```

**Key finding**: Significant improvement at short-medium sequences, but degrades at longer sequences (position 200+). This is likely due to increased L2 cache pressure from redundant RMSNorm reads competing with KV cache reads.

**Trade-off**: Best for interactive use cases with short contexts. May hurt throughput for long-context workloads.

## Optimization 2: Head-Based Work Distribution

**Status**: Implemented and tested - NOT VIABLE

**Approach**: Assign blocks to attention heads (5 blocks per Q head) so QKV + attention can proceed head-local without grid.sync between QKV and attention.

**Syncs eliminated**: 28 (1 per layer)

**Results**:
```
Position    Original    Optimized    Speedup
-------------------------------------------------
1           4.023ms     6.862ms      0.59x
10          4.051ms     6.892ms      0.59x
50          4.151ms     6.990ms      0.59x
100         4.272ms     7.039ms      0.61x
200         6.921ms     7.293ms      0.95x
-------------------------------------------------
Average     4.683ms     7.015ms      0.67x

Original:  213.5 tok/s
Optimized: 142.5 tok/s
Result:    -33% throughput (WORSE)
```

**Why it failed**:
1. QKV is memory-bound - all 82 blocks should participate for max memory parallelism
2. Only 16 leader blocks work during QKV phase - 66 blocks idle
3. Eliminating 1 sync per layer (~5-10us each x 28 = 140-280us) doesn't compensate for lost parallelism
4. The memory bandwidth loss dominates the sync savings

**Lesson**: Don't sacrifice block utilization to eliminate syncs. The sync overhead is small compared to the parallelism benefits.

## Optimization 3: Fused Phases

**Status**: Not yet implemented

**Approach**: Fuse adjacent phases:
- QKV projection + QK norm + RoPE
- O projection + residual + post-attention RMSNorm

**Expected syncs eliminated**: ~56 (2 per layer)

## Summary

| Optimization | Syncs Eliminated | Speedup | Status |
|-------------|-----------------|---------|--------|
| Redundant RMSNorm | 56 | 1.26x (avg), 1.42x (short seq) | Done |
| Head-Based Distribution | 28 | **0.67x (slower)** | Done - not viable |
| Fused Phases | 56 | TBD | Not started |

## Running the Experiments

```bash
# Redundant RMSNorm
python experiments/optimizations/redundant_rmsnorm/benchmark.py

# Compare all (baseline)
python experiments/optimizations/compare_all.py
```

## Files

- `redundant_rmsnorm/kernel.cu` - Optimized kernel with redundant RMSNorm
- `redundant_rmsnorm/benchmark.py` - Correctness verification and benchmarking
- `compare_all.py` - Baseline comparison script
