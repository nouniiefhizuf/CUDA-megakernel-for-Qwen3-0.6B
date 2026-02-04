# MegaQwen Development Log

A complete chronicle of building and optimizing a cooperative CUDA megakernel for Qwen3-0.6B inference.

**Final Result**: 530 tok/s decode on RTX 3090 (3.9x over HuggingFace, 1.5x over TensorRT-LLM at short contexts)

---

## Table of Contents

1. [The Goal](#the-goal)
2. [Architecture Overview](#architecture-overview)
3. [Framework Benchmarks](#framework-benchmarks)
4. [The Optimization Journey](#the-optimization-journey)
5. [Root Cause Analysis](#root-cause-analysis)
6. [What Worked vs What Didn't](#what-worked-vs-what-didnt)
7. [SASS/PTX Analysis](#sassptx-analysis)
8. [Architectural Ceiling](#architectural-ceiling)
9. [Lessons Learned](#lessons-learned)
10. [Future Directions](#future-directions)

---

## The Goal

Build a single cooperative CUDA kernel that fuses the entire Qwen3-0.6B forward pass for decode (batch=1). No intermediate memory writes, no kernel launch overhead - one kernel that processes input to output.

**Model**: Qwen3-0.6B
- 28 transformer layers
- 16 query heads, 8 KV heads (GQA)
- 1024 hidden dimension
- 128 head dimension
- 3072 MLP intermediate size

**Hardware**: NVIDIA RTX 3090
- 82 SMs, 10496 CUDA cores
- 936 GB/s memory bandwidth
- 35.6 TFLOPS FP32
- 24 GB VRAM

---

## Architecture Overview

The megakernel uses **cooperative groups** to synchronize across all 82 SMs:

```
┌─────────────────────────────────────────────────────────────┐
│                    MEGAKERNEL (82 blocks)                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 0-27 (loop):                                          │
│    ├─ RMSNorm (block 0 only)          → grid.sync()         │
│    ├─ QKV Projection (all blocks)     → grid.sync()         │
│    ├─ QK Norm + RoPE (all blocks)     → grid.sync()         │
│    ├─ Attention (16 blocks for 16 Q heads)                   │
│    │   └─ Other 66 blocks: L2 prefetch MLP weights           │
│    │                                  → grid.sync()         │
│    ├─ O Projection (all blocks)       → grid.sync()         │
│    ├─ Residual Add                                           │
│    ├─ RMSNorm (block 0 only)          → grid.sync()         │
│    ├─ Gate/Up Projection (all blocks) → grid.sync()         │
│    ├─ SiLU + Multiply                                        │
│    ├─ Down Projection (all blocks)    → grid.sync()         │
│    └─ Residual Add                                           │
├─────────────────────────────────────────────────────────────┤
│  LM Head (all blocks)                 → grid.sync()         │
│  Softmax (block 0)                                           │
└─────────────────────────────────────────────────────────────┘
```

**Synchronization**: ~225 `grid.sync()` calls per decode step (8 per layer x 28 layers + extras)

---

## Framework Benchmarks

### Long Context (200 tokens generated, ~22 token prompt)

| Framework | tok/s | Avg Power | tok/J | vs HuggingFace |
|-----------|-------|-----------|-------|----------------|
| **TensorRT-LLM** | **355** | 290W | **1.22** | **6.01x** |
| Megakernel | 158 | 205W | 0.77 | 2.68x |
| vLLM | 107 | 196W | 0.55 | 1.82x |
| SGLang | 107 | 210W | 0.51 | 1.81x |
| ExLlamaV2 | 98 | 197W | 0.50 | 1.66x |
| HuggingFace | 59 | 186W | 0.32 | 1.0x |
| llama.cpp | 50 | 195W | 0.26 | 0.85x |

### Short Context (100 tokens, 1 token prompt)

| Framework | tok/s | vs HuggingFace |
|-----------|-------|----------------|
| **Megakernel** | **530** | **3.91x** |
| TensorRT-LLM | 355 | 2.61x |
| HuggingFace | 136 | 1.0x |

**Key insight**: Megakernel beats TensorRT-LLM at short contexts (530 vs 355 tok/s) with zero compilation overhead.

### Position-Dependent Throughput

Decode throughput decreases with longer context due to attention reading more KV cache:

| Position | Before Optimization | After (L2 Prefetch) | Speedup |
|----------|---------------------|---------------------|---------|
| 1 | 242 | **525** | 2.17x |
| 10 | 241 | **527** | 2.19x |
| 50 | 229 | **500** | 2.18x |
| 100 | 175 | **472** | 2.70x |
| 200 | 142 | **422** | 2.97x |
| 300 | 139 | **382** | 2.75x |

---

## The Optimization Journey

### Phase 1: Baseline Implementation

Started with a straightforward cooperative kernel:
- One block per SM (82 blocks)
- 256 threads per block (8 warps)
- Standard `__ldg()` for cached weight loads
- `grid.sync()` between every phase

**Initial result**: ~170 tok/s

### Phase 2: Redundant RMSNorm

**Hypothesis**: RMSNorm requires all blocks to wait while block 0 computes. What if all blocks compute it redundantly?

```cuda
// Before: Only block 0 computes, others wait
if (block_id == 0) {
    compute_rmsnorm(...);
}
grid.sync();  // All blocks wait

// After: All blocks compute redundantly
compute_rmsnorm(...);  // No sync needed!
```

**Result**: +42% at short contexts (170 -> 215 tok/s), but degrades at long contexts due to L2 cache pressure from redundant weight reads.

**Syncs eliminated**: 56 (2 per layer x 28 layers)

### Phase 3: Block Divergence + L2 Prefetching

**Key insight**: During attention, only 16 blocks compute (one per Q head). The other 66 blocks are idle at `grid.sync()`.

**Solution**: Have idle blocks prefetch MLP weights into L2 cache:

```cuda
if (block_id < NUM_Q_HEADS) {
    // Compute attention
    compute_attention(block_id, ...);
} else {
    // Prefetch MLP weights using __ldg()
    prefetch_mlp_weights(block_id - NUM_Q_HEADS, ...);
}
grid.sync();
```

**Result**: **+2x speedup** (242 -> 530 tok/s at position 1)

This was the only optimization that provided substantial gains.

### Phase 4: 128-bit Vectorized Loads

**Hypothesis**: Loading 128 bits at once (uint4) instead of 64 bits (uint2) should improve memory coalescing.

```cuda
// Before: 64-bit loads
uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));

// After: 128-bit loads
uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
```

**SASS Analysis**:
```
Before: LDG.E.128 = 0 (no 128-bit loads)
After:  LDG.E.128 = 118 (128-bit loads present)
```

**Result**: +3.5% (1.904ms -> 1.838ms)

Minor improvement - we're not bandwidth-bound.

### Phase 5: Warp Producer/Consumer Specialization

**Hypothesis**: Dedicate some warps to prefetching while others compute.

```cuda
bool is_producer = (warp_id < NUM_PRODUCER_WARPS);
if (is_producer) {
    // Prefetch next phase's data
    prefetch_weights(...);
} else {
    // Consumer warps compute
    compute_matmul(...);
}
```

**Sweep Results**:

| Ratio (Producers:Consumers) | Avg tok/s |
|-----------------------------|-----------|
| **0:8** (all compute) | **509.9** |
| 1:7 | 510.8 |
| 2:6 | 489.9 |
| 3:5 | 498.7 |
| 4:4 | 478.6 |

**Result**: No improvement. All-compute is optimal.

**Why**: We're latency-bound by `grid.sync()`, not bandwidth-bound. Reducing compute parallelism hurts more than prefetching helps.

### Phase 6: Shared Memory Caching

**Hypothesis**: Cache `g_normalized` and `g_activations` (4KB each) in shared memory to avoid repeated global reads.

```cuda
__shared__ float smem_activations[1024];

// Load once at block start
for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    smem_activations[i] = g_activations[i];
}
__syncthreads();

// Use shared memory for all reads
sum += weight * smem_activations[k];
```

**Result**: 0% improvement (actually slight regression at some positions)

**Why**: L1/L2 cache already handles repeated reads effectively. The extra `__syncthreads()` overhead outweighs any theoretical savings.

### Phase 7: cp.async Double-Buffering

**Hypothesis**: Use async memory copies to overlap data loading with computation.

```cuda
#include <cuda_pipeline.h>

// Async load next tile while computing current
__pipeline_memcpy_async(smem_next, global_next, sizeof(float) * TILE_SIZE);
__pipeline_commit();

compute_tile(smem_current);  // Compute overlaps with load

__pipeline_wait_prior(0);
swap(smem_current, smem_next);
```

**Result**: +1% (1.827ms -> 1.808ms)

**Why**: Can't overlap enough compute. The matmul inner loops are too short to hide memory latency.

### Phase 8: Atomic Counter Sync (Explored)

**Hypothesis**: Replace `grid.sync()` with atomic counter spin-wait for lower latency.

```cuda
__device__ void atomic_barrier(int* counter, int* sense, int num_blocks) {
    __shared__ int local_sense;
    if (threadIdx.x == 0) {
        local_sense = *sense;
        int arrived = atomicAdd(counter, 1);
        if (arrived == num_blocks - 1) {
            *counter = 0;
            *sense = 1 - local_sense;  // Flip sense
        }
        while (*sense == local_sense) {}  // Spin-wait
    }
    __syncthreads();
}
```

**Status**: Inconclusive. Kernel compilation too slow for benchmarking.

---

## Root Cause Analysis

### Why We're Stuck at ~530 tok/s

After all optimizations, we discovered the fundamental bottleneck:

| Metric | Value |
|--------|-------|
| Effective memory bandwidth | ~47 GB/s |
| Peak memory bandwidth | 936 GB/s |
| **Utilization** | **5%** |
| grid.sync() calls per token | 140+ |
| Estimated sync latency | ~0.7 us each |

**We're using only 5% of available memory bandwidth.**

With 140 syncs at ~0.7us each = **~100us of pure synchronization overhead per token**.

### Why Memory Optimizations Don't Help

1. **Latency-bound, not bandwidth-bound**: Even 4x faster memory access won't help if we spend most time waiting at barriers.

2. **L1/L2 cache is already effective**: The GPU's cache hierarchy handles repeated reads well. Explicit shared memory caching adds overhead.

3. **Block-level prefetching already maxed out**: During attention, 66 blocks prefetch MLP weights. This is the optimal granularity - warp-level prefetching sacrifices compute parallelism.

### Cooperative vs CUDA Graph Overhead

| Approach | Time | Per-op Cost |
|----------|------|-------------|
| Cooperative + 225 grid.sync() | 167.3 us | 0.73 us/sync |
| CUDA graph (225 kernels) | 186.9 us | 0.83 us/kernel |
| Regular kernel launches | 347.5 us | 1.54 us/launch |

**Cooperative wins** by 19.7 us over CUDA graph. But the real benefit is avoiding ~2.7 MB of intermediate memory traffic per token.

---

## What Worked vs What Didn't

| Optimization | Impact | Why |
|--------------|--------|-----|
| Block divergence + L2 prefetch | **+2x** | Uses idle blocks during attention |
| Redundant RMSNorm | +42% (short ctx) | Eliminates 56 syncs |
| 128-bit vectorized loads | +3.5% | Better coalescing |
| cp.async double-buffering | +1% | Minor overlap |
| Warp producer/consumer | 0% | Reduces compute parallelism |
| Shared memory caching | 0% | L1/L2 already effective |
| Atomic counter sync | Unknown | Compilation too slow |

**Lesson**: Memory optimizations provide diminishing returns when latency-bound by synchronization.

---

## SASS/PTX Analysis

### Original Kernel (v1) - 64-bit Loads

```
LDG.E:              208  (32-bit loads)
LDG.E.U16.CONSTANT: 107  (16-bit scalar loads)
LDG.E.64.CONSTANT:   67  (64-bit loads)
LDG.E.CONSTANT:      60  (32-bit loads)
LDG.E.128:            0  (NO 128-bit loads)
STL (spills):         0  (no register spilling)
```

### Optimized Kernel (v2) - 128-bit Loads

```
LDG.E.128.CONSTANT:  66  (128-bit loads)
LDG.E.128:           52  (128-bit loads)
LDG.E.64.CONSTANT:   31  (64-bit loads)
LDG.E:               26  (32-bit loads)
STG.E.128:           29  (128-bit stores)
```

The 128-bit loads are present but provide only 3.5% improvement.

---

## Architectural Ceiling

**~530 tok/s is the architectural ceiling for batch=1 bf16 cooperative megakernels on RTX 3090.**

### Why This Is The Limit

1. **140+ grid.sync() calls** are inherent to the cooperative kernel approach where all SMs must reach barriers.

2. **Each sync takes ~0.7us** - this is hardware/driver overhead we can't eliminate.

3. **100us sync overhead per token** sets a floor on latency regardless of compute/memory optimizations.

### What Would Break The Ceiling

| Approach | Expected Gain | Difficulty |
|----------|---------------|------------|
| **INT4 Quantization** | ~4x | Medium |
| Non-cooperative architecture | Unknown | High (major rewrite) |
| Speculative decoding | ~2-4x | Medium |
| Larger batch size | Linear | N/A for single-user |

---

## Lessons Learned

### 1. Profile Before Optimizing

We assumed memory bandwidth was the issue. It wasn't. The 5% bandwidth utilization revealed we're latency-bound.

### 2. Cooperative Kernels Have Inherent Limits

`grid.sync()` overhead dominates at high sync counts. For 140+ syncs per token, this is unavoidable.

### 3. Block-Level Parallelism Matters

The +2x from L2 prefetching came from utilizing idle blocks during attention. This was the only substantial win.

### 4. Warp-Level Prefetching Doesn't Help Here

Trading compute for prefetch only makes sense when bandwidth-bound. We're not.

### 5. GPU Caches Are Better Than You Think

L1/L2 handle repeated reads effectively. Explicit shared memory caching often adds overhead without benefit.

### 6. SASS Analysis Is Essential

Looking at actual assembly revealed whether optimizations (like 128-bit loads) were actually being applied.

---

## Future Directions

### Quantization (Most Practical)

INT4 weights would reduce memory traffic 4x. Combined with existing optimizations, could reach 2000+ tok/s.

### Non-Cooperative Architecture

Eliminate `grid.sync()` entirely by having each SM process independent work. Major rewrite with unknown benefits.

### Speculative Decoding

Use a smaller draft model to propose multiple tokens, verify in parallel with the main model. Effectively increases batch size.

### Continuous Batching

Amortize sync overhead across multiple requests. Not applicable for single-user interactive use.

---

## Files Reference

```
csrc/megakernel/
├── fused_decode_ldg.cu          # Production kernel (L2 prefetch)
├── fused_decode_ldg_v2.cu       # 128-bit vectorized loads
├── fused_decode_ldg_v3.cu       # cp.async double-buffering
├── fused_decode_ldg_smem.cu     # Shared memory caching
├── fused_decode_atomic_sync.cu  # Atomic counter sync experiment
└── benchmark_*.py               # Various benchmarking scripts

experiments/
├── warp_sweep/                  # Producer/consumer ratio sweep
│   ├── sweep.py
│   └── kernel_warp_spec.cu
├── framework_bench/             # Framework comparisons
│   ├── benchmark_suite.py
│   ├── power_benchmark.py
│   └── quality_metrics.py
├── optimizations/
│   ├── redundant_rmsnorm/
│   └── head_based_distribution/
├── RESULTS.md                   # Complete benchmark data
└── sync_overhead.py             # Cooperative vs CUDA graph

docs/
├── ARCHITECTURE.md              # Kernel architecture details
└── MEMORY_ANALYSIS.md           # Memory bandwidth analysis
```

---

## Running Experiments

```bash
# Full benchmark suite
python experiments/framework_bench/full_benchmark.py

# Warp ratio sweep
python experiments/warp_sweep/sweep.py

# Sync overhead analysis
python experiments/sync_overhead.py

# Memory analysis (requires cuobjdump)
./analyze_sass.sh csrc/megakernel/fused_decode_ldg.cu

# Interactive chat
python chat.py
```

---

## Acknowledgments

This project explored the limits of cooperative megakernels for LLM inference. While we hit an architectural ceiling, the journey revealed fundamental insights about GPU synchronization overhead and the tradeoffs between memory bandwidth and latency.

The megakernel approach remains valuable for short-context, single-user, low-latency inference where it beats even TensorRT-LLM (530 vs 355 tok/s).

---

*Last updated: February 2026*
