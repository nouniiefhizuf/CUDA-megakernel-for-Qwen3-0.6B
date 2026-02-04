/**
 * Fused Decode with Atomic Counter Synchronization
 *
 * Optimization: Replace grid.sync() with atomic counter spin-wait pattern
 * for lower synchronization overhead.
 * 
 * Pattern: Sense-reversing barrier
 * - atomicAdd counter when block done
 * - Spin-wait until all blocks reach
 * - Use sense bit to avoid explicit resets
 */

#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Configuration
constexpr int ATOMIC_NUM_BLOCKS = 82;
constexpr int ATOMIC_BLOCK_SIZE = 256;
constexpr int ATOMIC_NUM_WARPS = ATOMIC_BLOCK_SIZE / WARP_SIZE;
constexpr float ATOMIC_RMS_EPS = 1e-6f;

// LM head
constexpr int ATOMIC_LM_NUM_BLOCKS = 1184;
constexpr int ATOMIC_LM_BLOCK_SIZE = 256;
constexpr int ATOMIC_VOCAB_SIZE = 151936;

// Number of sync points per layer (5) + embedding (1) = need counters
// Total sync points = 1 (embed) + num_layers * 5 + 1 (final norm) = 142 for 28 layers
// We'll use a rotating counter approach with sense bits
constexpr int MAX_SYNC_POINTS = 256;  // More than enough

struct AtomicSyncState {
    unsigned int counters[MAX_SYNC_POINTS];  // Arrival counters
    unsigned int sense[MAX_SYNC_POINTS];     // Sense bits for barrier
};

struct AtomicLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

// =============================================================================
// Atomic Barrier Implementation  
// =============================================================================

/**
 * Simple arrival-count barrier with generation tracking
 * 
 * Uses a monotonically increasing counter. Each barrier waits for the counter
 * to reach the expected value for that generation.
 * 
 * sync_state[0]: counter (incremented by each block at each barrier)
 * 
 * The key insight: we track which "generation" (barrier instance) we're on
 * using shared memory. Each block increments its local generation count.
 */

// Optimized atomic barrier with minimal fencing
// Uses release-acquire pattern: fence before arrive, fence after last block signals
__device__ __forceinline__ void atomic_barrier_v2(
    unsigned int* global_counter,
    int num_blocks
) {
    __syncthreads();  // Ensure all threads complete work before barrier
    
    if (threadIdx.x == 0) {
        // Memory fence before arrival to ensure prior writes are visible
        __threadfence();
        
        // Atomically increment and get our arrival count  
        unsigned int my_count = atomicAdd(global_counter, 1) + 1;
        
        // Calculate which generation we're in
        unsigned int my_gen = (my_count - 1) / num_blocks;
        
        // Target for this generation
        unsigned int target = (my_gen + 1) * num_blocks;
        
        // Spin until counter reaches target (with relaxed reads)
        unsigned int current;
        do {
            current = atomicAdd(global_counter, 0);
        } while (current < target);
        
        // No fence needed here - atomicAdd provides acquire semantics
    }
    
    __syncthreads();  // Broadcast completion to all threads
}

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float atomic_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float atomic_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// QKV Projection with Atomic Sync
// =============================================================================

__device__ void atomic_matvec_qkv(
    unsigned int* global_sync_var,
    int num_blocks,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Block 0 does RMSNorm
    if (block_id == 0) {
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_reduce[ATOMIC_NUM_WARPS];

        float local_sum_sq = 0.0f;

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_BLOCK_SIZE) {
            float v = __bfloat162float(__ldg(input + i));
            smem[i] = v;
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = atomic_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < ATOMIC_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = atomic_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + ATOMIC_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(norm_weight + i));
            g_normalized[i] = smem[i] * rstd * w;
        }
    }

    // Atomic sync instead of grid.sync()
    atomic_barrier_v2(global_sync_var, num_blocks);

    // QKV projection with vec4 and __ldg
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += ATOMIC_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < row_end) {
            const __nv_bfloat16* weight_row;
            float* output_ptr;

            if (m < Q_SIZE) {
                weight_row = q_weight + m * HIDDEN_SIZE;
                output_ptr = q_out + m;
            } else if (m < Q_SIZE + KV_SIZE) {
                weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
                output_ptr = k_out + (m - Q_SIZE);
            } else {
                weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
                output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
            }

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
                       __bfloat162float(w_ptr[1]) * g_normalized[k+1] +
                       __bfloat162float(w_ptr[2]) * g_normalized[k+2] +
                       __bfloat162float(w_ptr[3]) * g_normalized[k+3];
            }

            sum = atomic_warp_reduce_sum(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    // Atomic sync
    atomic_barrier_v2(global_sync_var, num_blocks);
}

// =============================================================================
// QK Norm + RoPE + KV Cache
// =============================================================================

__device__ void atomic_qk_norm_rope_cache(
    unsigned int* global_sync_var,
    int num_blocks,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    // Process Q heads
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += ATOMIC_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }
        sum_sq = atomic_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + ATOMIC_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
        }

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);

            if (i < HEAD_DIM/2) {
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            } else {
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
            }
        }
    }

    // Process K heads + cache
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += ATOMIC_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }
        sum_sq = atomic_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + ATOMIC_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
        }

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_final;
            if (i < HEAD_DIM/2) {
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            } else {
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            }
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }

    // Atomic sync
    atomic_barrier_v2(global_sync_var, num_blocks);
}

// =============================================================================
// Attention with Atomic Sync
// =============================================================================

__device__ void atomic_attention(
    unsigned int* global_sync_var,
    int num_blocks,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int ATTN_BLOCKS = NUM_Q_HEADS;

    if (block_id >= ATTN_BLOCKS) {
        // Prefetch weights while attention computes
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;

        float dummy = 0.0f;
        if (prefetch_block_id < num_prefetch_blocks / 3) {
            int elems_per_block = (Q_SIZE * HIDDEN_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = prefetch_block_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += ATOMIC_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(o_weight + start + i));
            }
        } else if (prefetch_block_id < 2 * num_prefetch_blocks / 3) {
            int adjusted_id = prefetch_block_id - num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += ATOMIC_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(gate_weight + start + i));
            }
        } else {
            int adjusted_id = prefetch_block_id - 2 * num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += ATOMIC_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(up_weight + start + i));
            }
        }

        __shared__ float s_dummy;
        if (threadIdx.x == 0) s_dummy = dummy;

        atomic_barrier_v2(global_sync_var, num_blocks);
        return;
    }

    __shared__ float s_max_score[ATOMIC_NUM_WARPS];
    __shared__ float s_sum_exp[ATOMIC_NUM_WARPS];
    __shared__ float s_out_acc[ATOMIC_NUM_WARPS][HEAD_DIM];

    int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

    for (int qh = head_start; qh < head_end; qh++) {
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        const float* q_head = q + qh * HEAD_DIM;
        float* out_head = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int pos = warp_id; pos < cache_len; pos += ATOMIC_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
            }
            score = atomic_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);

            float weight = expf(score - max_score);
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(v_pos + d));
            }
        }

        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        #pragma unroll
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
            s_out_acc[warp_id][d] = out_acc[j];
        }
        __syncthreads();

        if (warp_id == 0) {
            float global_max = s_max_score[0];
            for (int w = 1; w < ATOMIC_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < ATOMIC_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float scale = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * scale;

                    #pragma unroll
                    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                        final_out[j] += s_out_acc[w][d] * scale;
                    }
                }
            }

            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_head[d] = final_out[j] / total_sum_exp;
            }
        }
        __syncthreads();
    }

    atomic_barrier_v2(global_sync_var, num_blocks);
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP
// =============================================================================

__device__ void atomic_o_proj_postnorm_mlp(
    unsigned int* global_sync_var,
    int num_blocks,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ attn_out,
    float* __restrict__ g_residual,
    float* __restrict__ g_activations,
    float* __restrict__ g_mlp_intermediate,
    __nv_bfloat16* __restrict__ hidden_out
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // O Projection + Residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += ATOMIC_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* o_row = o_weight + m * Q_SIZE;

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < Q_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * attn_out[k] +
                       __bfloat162float(w_ptr[1]) * attn_out[k+1] +
                       __bfloat162float(w_ptr[2]) * attn_out[k+2] +
                       __bfloat162float(w_ptr[3]) * attn_out[k+3];
            }

            sum = atomic_warp_reduce_sum(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    atomic_barrier_v2(global_sync_var, num_blocks);

    // Post-attention RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[ATOMIC_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_BLOCK_SIZE) {
            float v = g_activations[i];
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = atomic_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < ATOMIC_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = atomic_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + ATOMIC_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            g_activations[i] = g_residual[i] * rstd * w;
        }
    }

    atomic_barrier_v2(global_sync_var, num_blocks);

    // Gate + Up + SiLU
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    for (int m_base = int_start; m_base < int_end; m_base += ATOMIC_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < int_end) {
            const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
            const __nv_bfloat16* up_row = up_weight + m * HIDDEN_SIZE;

            float gate_sum = 0.0f, up_sum = 0.0f;

            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_row + k));
                uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

                gate_sum += __bfloat162float(g_ptr[0]) * g_activations[k] +
                            __bfloat162float(g_ptr[1]) * g_activations[k+1] +
                            __bfloat162float(g_ptr[2]) * g_activations[k+2] +
                            __bfloat162float(g_ptr[3]) * g_activations[k+3];

                up_sum += __bfloat162float(u_ptr[0]) * g_activations[k] +
                          __bfloat162float(u_ptr[1]) * g_activations[k+1] +
                          __bfloat162float(u_ptr[2]) * g_activations[k+2] +
                          __bfloat162float(u_ptr[3]) * g_activations[k+3];
            }

            gate_sum = atomic_warp_reduce_sum(gate_sum);
            up_sum = atomic_warp_reduce_sum(up_sum);

            if (lane_id == 0) {
                g_mlp_intermediate[m] = atomic_silu(gate_sum) * up_sum;
            }
        }
    }

    atomic_barrier_v2(global_sync_var, num_blocks);

    // Down projection + residual
    for (int m_base = hid_start; m_base < hid_end; m_base += ATOMIC_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* down_row = down_weight + m * INTERMEDIATE_SIZE;

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
                uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

                sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k] +
                       __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k+1] +
                       __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k+2] +
                       __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k+3];
            }

            sum = atomic_warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
            }
        }
    }

    atomic_barrier_v2(global_sync_var, num_blocks);
}

// =============================================================================
// Main Kernel with Atomic Sync
// =============================================================================

__global__ void __launch_bounds__(ATOMIC_BLOCK_SIZE, 1)
atomic_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const AtomicLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden_buffer,
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    unsigned int* __restrict__ global_sync_var,  // Single rolling counter
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Embedding lookup with __ldg
    const __nv_bfloat16* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * ATOMIC_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * ATOMIC_BLOCK_SIZE) {
        hidden_buffer[i] = __ldg(embed_row + i);
    }
    
    // Initial sync after embedding
    atomic_barrier_v2(global_sync_var, num_blocks);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const AtomicLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        atomic_matvec_qkv(
            global_sync_var, num_blocks,
            hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v
        );

        atomic_qk_norm_rope_cache(
            global_sync_var, num_blocks,
            g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len
        );

        atomic_attention(
            global_sync_var, num_blocks,
            g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale,
            w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight
        );

        atomic_o_proj_postnorm_mlp(
            global_sync_var, num_blocks,
            w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer
        );
    }

    // Final RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[ATOMIC_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            g_activations[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = atomic_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < ATOMIC_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = atomic_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + ATOMIC_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
}

// =============================================================================
// LM Head Kernels (same as LDG version)
// =============================================================================

__global__ void atomic_lm_head_logits(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ logits
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (ATOMIC_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, ATOMIC_VOCAB_SIZE);

    for (int m = row_start + warp_id; m < row_end; m += ATOMIC_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;

        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = atomic_warp_reduce_sum(sum);

        if (lane_id == 0) {
            logits[m] = sum;
        }
    }
}

__global__ void atomic_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += ATOMIC_LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (ATOMIC_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, ATOMIC_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += ATOMIC_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;

        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = atomic_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[ATOMIC_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[ATOMIC_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < ATOMIC_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < ATOMIC_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_val;
            block_max_idxs[blockIdx.x] = max_idx;
        }
    }
}

__global__ void atomic_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
    __shared__ float s_max_vals[1024];
    __shared__ int s_max_idxs[1024];

    int tid = threadIdx.x;

    float local_max = -INFINITY;
    int local_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_max_vals[i];
        if (val > local_max) {
            local_max = val;
            local_idx = block_max_idxs[i];
        }
    }

    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_vals[tid + s] > s_max_vals[tid]) {
                s_max_vals[tid] = s_max_vals[tid + s];
                s_max_idxs[tid] = s_max_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output_token = s_max_idxs[0];
    }
}

// =============================================================================
// Launch Functions
// =============================================================================

extern "C" void launch_atomic_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const AtomicLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* global_sync_var,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    // Reset sync counter before kernel launch
    cudaMemsetAsync(global_sync_var, 0, sizeof(unsigned int), stream);
    
    // Launch main decode kernel (NOT cooperative - uses atomic sync instead)
    atomic_decode_kernel<<<ATOMIC_NUM_BLOCKS, ATOMIC_BLOCK_SIZE, 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        layer_weights,
        (const __nv_bfloat16*)final_norm_weight,
        (const __nv_bfloat16*)cos_table,
        (const __nv_bfloat16*)sin_table,
        (__nv_bfloat16*)k_cache,
        (__nv_bfloat16*)v_cache,
        (__nv_bfloat16*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        (unsigned int*)global_sync_var,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale
    );

    atomic_lm_head_phase1<<<ATOMIC_LM_NUM_BLOCKS, ATOMIC_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    atomic_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        ATOMIC_LM_NUM_BLOCKS
    );
}

extern "C" void launch_atomic_decode_with_logits(
    int input_token_id,
    int* output_token_id,
    float* logits_output,
    const void* embed_weight,
    const AtomicLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* global_sync_var,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    // Reset sync counter before kernel launch
    cudaMemsetAsync(global_sync_var, 0, sizeof(unsigned int), stream);
    
    atomic_decode_kernel<<<ATOMIC_NUM_BLOCKS, ATOMIC_BLOCK_SIZE, 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        layer_weights,
        (const __nv_bfloat16*)final_norm_weight,
        (const __nv_bfloat16*)cos_table,
        (const __nv_bfloat16*)sin_table,
        (__nv_bfloat16*)k_cache,
        (__nv_bfloat16*)v_cache,
        (__nv_bfloat16*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        (unsigned int*)global_sync_var,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale
    );

    atomic_lm_head_logits<<<ATOMIC_LM_NUM_BLOCKS, ATOMIC_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        logits_output
    );

    atomic_lm_head_phase1<<<ATOMIC_LM_NUM_BLOCKS, ATOMIC_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    atomic_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        ATOMIC_LM_NUM_BLOCKS
    );
}
