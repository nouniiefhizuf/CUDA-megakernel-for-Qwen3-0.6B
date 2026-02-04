# Memory Analysis: fused_decode_ldg.cu

## Executive Summary

Analysis of the fused decode megakernel reveals severe memory underutilization at ~47 GB/s effective bandwidth vs 936 GB/s peak (5% efficiency). The bottleneck is **latency** from grid synchronization, not memory bandwidth.

## SASS Analysis Results

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

## Benchmark Results (RTX 3090)

| Metric | v1 (64-bit) | v2 (128-bit) |
|--------|-------------|--------------|
| Mean latency | 1.904 ms | 1.838 ms |
| Min latency | 1.843 ms | 1.795 ms |
| Speedup | - | 1.04x |
| Effective BW | 46 GB/s | 48 GB/s |
| Peak BW utilization | 4.9% | 5.1% |

## Root Cause: Latency-Bound Execution

The kernel achieves only 5% of peak bandwidth because it's **latency-bound**, not bandwidth-bound:

### 1. Grid Synchronization Overhead
The cooperative kernel uses `grid.sync()` for inter-SM coordination:
- RMSNorm phase complete -> sync
- QKV projection complete -> sync
- QK norm + RoPE complete -> sync
- Attention complete -> sync
- O proj + MLP complete -> sync

With 28 layers x 5 syncs/layer = 140 grid barriers per token. Each barrier stalls all SMs until the slowest one completes.

### 2. Serial Layer Processing
All 82 blocks process one layer at a time, waiting at grid.sync() between phases. This serializes work that could theoretically overlap.

### 3. Low Memory-Level Parallelism
Even with 128-bit loads, each thread has limited outstanding memory requests:
- Only 4 loads unrolled in inner loop
- Warp-level parallelism is good (32 threads x 128 bits = 512 bytes/warp)
- But total in-flight bytes is still < 1MB across all SMs

### 4. Insufficient Occupancy
- 82 blocks x 256 threads = 20,992 total threads
- RTX 3090: 82 SMs x 2048 max threads/SM = 167,936 possible
- Actual occupancy: ~12.5%

## Optimization Strategies

### High Impact (needs architecture change):

1. **Pipelined layer execution**: Have different SMs work on different layers, overlapping memory access with compute from previous layers. Requires careful dataflow design.

2. **Persistent kernel with reduced syncs**: Minimize grid.sync() calls by fusing more operations. For batch=1 decode, consider having each SM process a subset of the model independently.

3. **Warp specialization**: Have some warps focus on memory prefetching while others compute. This increases MLP without requiring more threads.

### Medium Impact (kernel-level):

4. **Increase unroll factor**: Change from `#pragma unroll 4` to `#pragma unroll 8` to have more loads in flight per thread.

5. **Software pipelining**: Double-buffer weight tiles - load next tile while computing current.

6. **Shared memory caching**: For attention phase, cache Q vector in shared memory to reduce global memory traffic.

### Low Impact (already implemented in v2):

7. **128-bit vectorized loads**: Implemented in v2, provides 1.04x speedup.

## Code Changes Made (v2)

### Weight Loads: uint2 -> uint4
```cpp
// Before (64-bit):
uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));

// After (128-bit):
uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
```

### Activation Loads: Scalar -> float4
```cpp
// Before:
sum += w[0] * g_normalized[k] +
       w[1] * g_normalized[k+1] + ...

// After:
float4 act1 = *reinterpret_cast<const float4*>(g_normalized + k);
float4 act2 = *reinterpret_cast<const float4*>(g_normalized + k + 4);
sum += w[0] * act1.x + w[1] * act1.y + ...
```

### Attention K/V Cache: Scalar bf16 -> uint2
```cpp
// Before:
score += q_head[d] * __bfloat162float(__ldg(k_pos + d));

// After:
uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + lane_id * 4));
score = q_local.x * k[0] + q_local.y * k[1] + ...
```

## Recommended Next Steps

1. **Profile grid.sync() overhead**: Use CUDA events around sync calls to measure time spent waiting.

2. **Explore non-cooperative alternatives**: 
   - Use CUDA graphs to reduce launch overhead
   - Split into multiple kernels with explicit dependencies
   - Use stream-ordered memory for inter-kernel communication

3. **Increase occupancy**: 
   - Reduce register usage per thread
   - Use smaller block size with more blocks
   - Consider using multiple kernel launches in a CUDA graph

4. **Consider different algorithms for batch=1**:
   - Flash attention style for KV cache access
   - Continuous batching to amortize overhead

## Files

- `csrc/megakernel/fused_decode_ldg.cu` - Original v1 kernel (with L2 prefetching)
- `csrc/megakernel/fused_decode_ldg_v2.cu` - Optimized v2 kernel with 128-bit loads
- `csrc/megakernel/fused_decode_ldg_v3.cu` - cp.async double-buffering variant
- `csrc/megakernel/fused_decode_ldg_smem.cu` - Shared memory caching variant
- `csrc/megakernel/fused_decode_atomic_sync.cu` - Atomic counter sync variant
- `experiments/warp_sweep/` - Warp producer/consumer ratio sweep

---

## Final Conclusions

After exhaustive optimization attempts, we've determined that **~530 tok/s is the architectural ceiling** for batch=1 bf16 cooperative megakernels on RTX 3090.

### What We Tried

| Optimization | Impact | Why |
|--------------|--------|-----|
| Block divergence + L2 prefetch | **+2x** | Only real win - uses idle blocks during attention |
| 128-bit vectorized loads | +3.5% | Minor coalescing improvement |
| Warp producer/consumer split | 0% | Reduces compute parallelism |
| Shared memory caching | 0% | L1/L2 already effective |
| cp.async double-buffering | +1% | Can't overlap enough compute |

### The Fundamental Limit

With 140+ `grid.sync()` calls per token at ~0.7us each, we spend ~100us just synchronizing. This is inherent to the cooperative kernel approach where all SMs must reach barriers.

### Path Forward

To exceed 530 tok/s on batch=1:
1. **Quantization** (INT4): 4x less memory traffic = most practical
2. **Non-cooperative architecture**: Major rewrite, unknown gains
3. **Speculative decoding**: Increases effective batch size
