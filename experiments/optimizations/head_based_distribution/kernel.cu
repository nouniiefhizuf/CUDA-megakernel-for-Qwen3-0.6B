/**
 * Head-Based Distribution Kernel Optimization
 *
 * Key optimization: Assign blocks to Q heads (82 blocks / 16 heads ≈ 5 blocks per head).
 * Only the leader block in each head group performs QKV/attention work.
 * This eliminates the grid.sync between QKV and attention phases.
 *
 * Block assignment:
 * - blocks [0-4]   -> Q head 0 (KV head 0)
 * - blocks [5-9]   -> Q head 1 (KV head 0)
 * - blocks [10-14] -> Q head 2 (KV head 1)
 * - ... and so on
 *
 * With 82 blocks and 16 Q heads:
 * - First 80 blocks: 5 blocks per head (16 heads * 5 = 80)
 * - Last 2 blocks: assigned to heads 0 and 1
 *
 * Trade-off:
 * - Eliminates 1 grid.sync() per layer (between QKV and attention)
 * - Only leader blocks do QKV work, others idle during that phase
 * - Attention uses all blocks in the head group for parallelism
 *
 * Expected benefit: 28 layers * 1 sync = 28 grid.sync() calls eliminated
 */

#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Configuration
constexpr int HBD_NUM_BLOCKS = 82;
constexpr int HBD_BLOCK_SIZE = 256;
constexpr int HBD_NUM_WARPS = HBD_BLOCK_SIZE / WARP_SIZE;
constexpr float HBD_RMS_EPS = 1e-6f;

// Head-based distribution constants
constexpr int HBD_BLOCKS_PER_HEAD = 5;  // 80 blocks for 16 heads, 2 extra assigned to first 2 heads
constexpr int HBD_EXTRA_BLOCKS = HBD_NUM_BLOCKS - (NUM_Q_HEADS * HBD_BLOCKS_PER_HEAD);  // 82 - 80 = 2

// LM head
constexpr int HBD_LM_NUM_BLOCKS = 1184;
constexpr int HBD_LM_BLOCK_SIZE = 256;
constexpr int HBD_VOCAB_SIZE = 151936;

struct HBDLayerWeights {
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
// Helpers
// =============================================================================

__device__ __forceinline__ float hbd_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float hbd_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Block-to-Head Mapping
// =============================================================================

/**
 * Get the Q head index for a given block.
 * Blocks 0-4 -> head 0, 5-9 -> head 1, etc.
 * Blocks 80 and 81 are assigned to heads 0 and 1 respectively.
 */
__device__ __forceinline__ int get_q_head_for_block(int block_id) {
    if (block_id < NUM_Q_HEADS * HBD_BLOCKS_PER_HEAD) {
        return block_id / HBD_BLOCKS_PER_HEAD;
    }
    // Extra blocks assigned to first heads
    return block_id - NUM_Q_HEADS * HBD_BLOCKS_PER_HEAD;
}

/**
 * Check if this block is the leader for its head group.
 * Leader blocks perform QKV computation.
 */
__device__ __forceinline__ bool is_leader_block(int block_id) {
    if (block_id < NUM_Q_HEADS * HBD_BLOCKS_PER_HEAD) {
        return (block_id % HBD_BLOCKS_PER_HEAD) == 0;
    }
    // Extra blocks are not leaders
    return false;
}

/**
 * Get the first block ID for a given Q head.
 */
__device__ __forceinline__ int get_first_block_for_head(int q_head) {
    return q_head * HBD_BLOCKS_PER_HEAD;
}

/**
 * Get the number of blocks assigned to a given Q head.
 */
__device__ __forceinline__ int get_blocks_for_head(int q_head) {
    int base_blocks = HBD_BLOCKS_PER_HEAD;
    // First HBD_EXTRA_BLOCKS heads get an extra block
    if (q_head < HBD_EXTRA_BLOCKS) {
        return base_blocks + 1;
    }
    return base_blocks;
}

/**
 * Get the local block index within a head group.
 */
__device__ __forceinline__ int get_local_block_idx(int block_id, int q_head) {
    int first_block = get_first_block_for_head(q_head);
    if (block_id >= NUM_Q_HEADS * HBD_BLOCKS_PER_HEAD) {
        // This is an extra block
        return HBD_BLOCKS_PER_HEAD;
    }
    return block_id - first_block;
}

// =============================================================================
// RMSNorm - All blocks compute redundantly (from redundant_rmsnorm optimization)
// =============================================================================

__device__ void hbd_redundant_rmsnorm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* rstd_out  // Output the rstd for use in matvec
) {
    __shared__ float smem[HIDDEN_SIZE];
    __shared__ float smem_reduce[HBD_NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_BLOCK_SIZE) {
        float v = __bfloat162float(__ldg(input + i));
        smem[i] = v;
        g_residual[i] = v;
        local_sum_sq += v * v;
    }

    local_sum_sq = hbd_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < HBD_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = hbd_warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + HBD_RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    *rstd_out = rstd;

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(weight + i));
        g_normalized[i] = smem[i] * rstd * w;
    }
}

// =============================================================================
// Head-Based QKV Computation
// Only leader blocks compute Q for their head, and K/V for corresponding KV head
// =============================================================================

__device__ void hbd_compute_qkv_for_head(
    int q_head,
    const float* __restrict__ g_normalized,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Compute Q for this head (HEAD_DIM elements)
    float* q_head_out = q_out + q_head * HEAD_DIM;
    const __nv_bfloat16* q_head_weight = q_weight + q_head * HEAD_DIM * HIDDEN_SIZE;

    // Each warp computes one Q element
    for (int d = warp_id; d < HEAD_DIM; d += HBD_NUM_WARPS) {
        const __nv_bfloat16* w_row = q_head_weight + d * HIDDEN_SIZE;

        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
                   __bfloat162float(w_ptr[1]) * g_normalized[k+1] +
                   __bfloat162float(w_ptr[2]) * g_normalized[k+2] +
                   __bfloat162float(w_ptr[3]) * g_normalized[k+3];
        }

        sum = hbd_warp_reduce_sum(sum);
        if (lane_id == 0) {
            q_head_out[d] = sum;
        }
    }

    // Compute K and V for the corresponding KV head
    // Only the first Q head in each KV group computes K/V
    int kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);
    bool is_first_q_for_kv = (q_head % (NUM_Q_HEADS / NUM_KV_HEADS)) == 0;

    if (is_first_q_for_kv) {
        float* k_head_out = k_out + kv_head * HEAD_DIM;
        float* v_head_out = v_out + kv_head * HEAD_DIM;
        const __nv_bfloat16* k_head_weight = k_weight + kv_head * HEAD_DIM * HIDDEN_SIZE;
        const __nv_bfloat16* v_head_weight = v_weight + kv_head * HEAD_DIM * HIDDEN_SIZE;

        for (int d = warp_id; d < HEAD_DIM; d += HBD_NUM_WARPS) {
            const __nv_bfloat16* k_row = k_head_weight + d * HIDDEN_SIZE;
            const __nv_bfloat16* v_row = v_head_weight + d * HIDDEN_SIZE;

            float k_sum = 0.0f, v_sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_row + k));
                uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_row + k));
                __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(&k_u2);
                __nv_bfloat16* v_ptr = reinterpret_cast<__nv_bfloat16*>(&v_u2);

                k_sum += __bfloat162float(k_ptr[0]) * g_normalized[k] +
                         __bfloat162float(k_ptr[1]) * g_normalized[k+1] +
                         __bfloat162float(k_ptr[2]) * g_normalized[k+2] +
                         __bfloat162float(k_ptr[3]) * g_normalized[k+3];

                v_sum += __bfloat162float(v_ptr[0]) * g_normalized[k] +
                         __bfloat162float(v_ptr[1]) * g_normalized[k+1] +
                         __bfloat162float(v_ptr[2]) * g_normalized[k+2] +
                         __bfloat162float(v_ptr[3]) * g_normalized[k+3];
            }

            k_sum = hbd_warp_reduce_sum(k_sum);
            v_sum = hbd_warp_reduce_sum(v_sum);
            if (lane_id == 0) {
                k_head_out[d] = k_sum;
                v_head_out[d] = v_sum;
            }
        }
    }
}

// =============================================================================
// QK Norm + RoPE for a single head
// =============================================================================

__device__ void hbd_qk_norm_rope_for_head(
    int q_head,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_pos,
    const __nv_bfloat16* __restrict__ sin_pos,
    __nv_bfloat16* __restrict__ k_cache_head,
    __nv_bfloat16* __restrict__ v_cache_head
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);

    // Process Q head with warp 0
    if (warp_id == 0) {
        float* q_head_ptr = q + q_head * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head_ptr[i] * q_head_ptr[i];
        }
        sum_sq = hbd_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + HBD_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            q_local[j] = q_head_ptr[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
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
                q_head_ptr[i] = q_local[j] * cos_v - pair_v * sin_v;
            } else {
                q_head_ptr[i] = pair_v * sin_v + q_local[j] * cos_v;
            }
        }
    }

    // Process K head with warp 1 (if this Q head is responsible for K)
    bool is_first_q_for_kv = (q_head % (NUM_Q_HEADS / NUM_KV_HEADS)) == 0;
    if (is_first_q_for_kv && warp_id == 1) {
        float* k_head_ptr = k + kv_head * HEAD_DIM;
        const float* v_head_ptr = v + kv_head * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head_ptr[i] * k_head_ptr[i];
        }
        sum_sq = hbd_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + HBD_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            k_local[j] = k_head_ptr[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
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
            k_head_ptr[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head_ptr[i]);
        }
    }

    __syncthreads();
}

// =============================================================================
// Head-Based Attention (leader block computes full attention for its head)
// =============================================================================

__device__ void hbd_attention_for_head(
    int q_head,
    bool is_leader,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    // Only leader blocks compute attention for their head
    // This avoids the complexity of cross-block reduction
    if (!is_leader) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);

    __shared__ float s_max_score[HBD_NUM_WARPS];
    __shared__ float s_sum_exp[HBD_NUM_WARPS];
    __shared__ float s_out_acc[HBD_NUM_WARPS][HEAD_DIM];

    const float* q_head_ptr = q + q_head * HEAD_DIM;
    float* out_head = attn_out + q_head * HEAD_DIM;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each warp processes a subset of cache positions
    for (int pos = warp_id; pos < cache_len; pos += HBD_NUM_WARPS) {
        const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
        const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

        // Q @ K
        float score = 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            score += q_head_ptr[d] * __bfloat162float(__ldg(k_pos + d));
        }
        score = hbd_warp_reduce_sum(score) * attn_scale;
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

    // Store each warp's partial results
    if (lane_id == 0) {
        s_max_score[warp_id] = max_score;
        s_sum_exp[warp_id] = sum_exp;
    }
    #pragma unroll
    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
        s_out_acc[warp_id][d] = out_acc[j];
    }
    __syncthreads();

    // Warp 0 combines results from all warps
    if (warp_id == 0) {
        float global_max = s_max_score[0];
        for (int w = 1; w < HBD_NUM_WARPS; w++) {
            if (s_max_score[w] > -INFINITY) {
                global_max = fmaxf(global_max, s_max_score[w]);
            }
        }

        float total_sum_exp = 0.0f;
        float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int w = 0; w < HBD_NUM_WARPS; w++) {
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

// =============================================================================
// O Projection + Residual + PostNorm + MLP (standard distribution)
// =============================================================================

__device__ void hbd_o_proj_postnorm_mlp(
    cg::grid_group& grid,
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
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // O Projection + Residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += HBD_NUM_WARPS) {
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

            sum = hbd_warp_reduce_sum(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    grid.sync();

    // Post-attention RMSNorm (redundant computation)
    __shared__ float smem[HIDDEN_SIZE];
    __shared__ float smem_reduce[HBD_NUM_WARPS];

    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_BLOCK_SIZE) {
        float v = g_activations[i];
        smem[i] = v;
        g_residual[i] = v;
        local_sum_sq += v * v;
    }

    local_sum_sq = hbd_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
        smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < HBD_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = hbd_warp_reduce_sum(sum);
        if (lane_id == 0) {
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + HBD_RMS_EPS);
        }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_BLOCK_SIZE) {
        float w = __bfloat162float(__ldg(post_norm_weight + i));
        g_activations[i] = smem[i] * rstd * w;
    }

    // Gate + Up + SiLU
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    for (int m_base = int_start; m_base < int_end; m_base += HBD_NUM_WARPS) {
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

                float n0 = smem[k] * rstd * __bfloat162float(__ldg(post_norm_weight + k));
                float n1 = smem[k+1] * rstd * __bfloat162float(__ldg(post_norm_weight + k+1));
                float n2 = smem[k+2] * rstd * __bfloat162float(__ldg(post_norm_weight + k+2));
                float n3 = smem[k+3] * rstd * __bfloat162float(__ldg(post_norm_weight + k+3));

                gate_sum += __bfloat162float(g_ptr[0]) * n0 +
                            __bfloat162float(g_ptr[1]) * n1 +
                            __bfloat162float(g_ptr[2]) * n2 +
                            __bfloat162float(g_ptr[3]) * n3;

                up_sum += __bfloat162float(u_ptr[0]) * n0 +
                          __bfloat162float(u_ptr[1]) * n1 +
                          __bfloat162float(u_ptr[2]) * n2 +
                          __bfloat162float(u_ptr[3]) * n3;
            }

            gate_sum = hbd_warp_reduce_sum(gate_sum);
            up_sum = hbd_warp_reduce_sum(up_sum);

            if (lane_id == 0) {
                g_mlp_intermediate[m] = hbd_silu(gate_sum) * up_sum;
            }
        }
    }

    grid.sync();

    // Down projection + residual
    for (int m_base = hid_start; m_base < hid_end; m_base += HBD_NUM_WARPS) {
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

            sum = hbd_warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Main Kernel
// =============================================================================

__global__ void __launch_bounds__(HBD_BLOCK_SIZE, 1)
hbd_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const HBDLayerWeights* __restrict__ layer_weights,
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
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Embedding lookup
    const __nv_bfloat16* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * HBD_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * HBD_BLOCK_SIZE) {
        hidden_buffer[i] = __ldg(embed_row + i);
    }
    grid.sync();

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    // Get head assignment for this block
    int my_q_head = get_q_head_for_block(block_id);
    bool am_leader = is_leader_block(block_id);
    int local_block_idx = get_local_block_idx(block_id, my_q_head);
    int blocks_for_my_head = get_blocks_for_head(my_q_head);

    for (int layer = 0; layer < num_layers; layer++) {
        const HBDLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        // =====================================================================
        // Phase 1: RMSNorm (all blocks compute redundantly)
        // =====================================================================
        float rstd;
        hbd_redundant_rmsnorm(hidden_buffer, w.input_layernorm_weight,
                              g_normalized, g_residual, &rstd);
        // No grid.sync needed - all blocks have same normalized values

        // =====================================================================
        // Phase 2: QKV projection (only leader blocks)
        // Each leader computes Q for its head and K/V for corresponding KV head
        // =====================================================================
        if (am_leader) {
            hbd_compute_qkv_for_head(my_q_head, g_normalized,
                                     w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
                                     g_q, g_k, g_v);
        }

        // Need sync here for QKV to be visible to all blocks in head group
        grid.sync();

        // =====================================================================
        // Phase 3: QK Norm + RoPE + Cache update (leader blocks only)
        // =====================================================================
        int kv_head = my_q_head / (NUM_Q_HEADS / NUM_KV_HEADS);
        const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
        const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

        if (am_leader) {
            __nv_bfloat16* k_cache_head = layer_k_cache + kv_head * max_seq_len * HEAD_DIM + position * HEAD_DIM;
            __nv_bfloat16* v_cache_head = layer_v_cache + kv_head * max_seq_len * HEAD_DIM + position * HEAD_DIM;

            hbd_qk_norm_rope_for_head(my_q_head, g_q, g_k, g_v,
                                      w.q_norm_weight, w.k_norm_weight,
                                      cos_pos, sin_pos,
                                      k_cache_head, v_cache_head);
        }

        // Need sync for rope'd Q values and updated K cache
        grid.sync();

        // =====================================================================
        // Phase 4: Attention (leader blocks compute full attention for their head)
        // =====================================================================
        hbd_attention_for_head(my_q_head, am_leader,
                               g_q, layer_k_cache, layer_v_cache, g_attn_out,
                               cache_len, max_seq_len, attn_scale);

        // Need sync for attention output before O projection
        grid.sync();

        // =====================================================================
        // Phase 5: O Projection + PostNorm + MLP (standard distribution)
        // =====================================================================
        hbd_o_proj_postnorm_mlp(grid, w.o_proj_weight, w.post_attn_layernorm_weight,
                                w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                                g_attn_out, g_residual, g_activations, g_mlp_intermediate,
                                hidden_buffer);
    }

    // Final RMSNorm (all blocks compute redundantly)
    {
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_reduce[HBD_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            smem[i] = v;
            g_activations[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = hbd_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < HBD_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = hbd_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + HBD_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = smem[i] * rstd * wt;
        }
    }
}

// =============================================================================
// LM Head (unchanged)
// =============================================================================

__global__ void hbd_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += HBD_LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (HBD_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, HBD_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += HBD_LM_BLOCK_SIZE / WARP_SIZE) {
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
        sum = hbd_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[HBD_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[HBD_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < HBD_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < HBD_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

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

__global__ void hbd_lm_head_phase2(
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
// Launch function
// =============================================================================

extern "C" void launch_hbd_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const HBDLayerWeights* layer_weights,
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
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    void* kernel_args[] = {
        (void*)&input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&num_layers,
        (void*)&position,
        (void*)&cache_len,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    cudaLaunchCooperativeKernel(
        (void*)hbd_decode_kernel,
        dim3(HBD_NUM_BLOCKS),
        dim3(HBD_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    hbd_lm_head_phase1<<<HBD_LM_NUM_BLOCKS, HBD_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    hbd_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        HBD_LM_NUM_BLOCKS
    );
}
