/**
 * Fused Decode with warp specialization
 *
 * This kernel variant tests producer/consumer warp ratios.
 * NUM_PRODUCER_WARPS is set by compile-time define.
 */

#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Configuration
constexpr int LDG_NUM_BLOCKS = 82;
constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_RMS_EPS = 1e-6f;

// Warp specialization - set via compile define, default 0 (all compute)
#ifndef NUM_PRODUCER_WARPS
#define NUM_PRODUCER_WARPS 0
#endif

constexpr int PRODUCER_WARPS = NUM_PRODUCER_WARPS;
constexpr int CONSUMER_WARPS = LDG_NUM_WARPS - PRODUCER_WARPS;

// LM head
constexpr int LDG_LM_NUM_BLOCKS = 1184;
constexpr int LDG_LM_BLOCK_SIZE = 256;
constexpr int LDG_VOCAB_SIZE = 151936;

struct WarpSpecLayerWeights {
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

__device__ __forceinline__ float ws_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float ws_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Prefetch helper for producer warps
__device__ __forceinline__ void ws_prefetch_range(
    const __nv_bfloat16* __restrict__ ptr,
    int start,
    int count,
    int stride
) {
    float dummy = 0.0f;
    for (int i = start; i < start + count; i += stride) {
        dummy += __bfloat162float(__ldg(ptr + i));
    }
    // Prevent optimization
    if (dummy == -12345.0f) {
        asm volatile("" ::: "memory");
    }
}

// =============================================================================
// QKV Matvec with warp specialization
// =============================================================================

__device__ void ws_matvec_qkv(
    cg::grid_group& grid,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    // Next phase weights for prefetch
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    int position
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    bool is_producer = (PRODUCER_WARPS > 0) && (warp_id < PRODUCER_WARPS);
    int consumer_warp_id = warp_id - PRODUCER_WARPS;
    int active_warps = (PRODUCER_WARPS > 0) ? CONSUMER_WARPS : LDG_NUM_WARPS;

    // Block 0 does RMSNorm (all warps participate for speed)
    if (block_id == 0) {
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __bfloat162float(__ldg(input + i));
            smem[i] = v;
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = ws_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ws_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(norm_weight + i));
            g_normalized[i] = smem[i] * rstd * w;
        }
    }

    grid.sync();

    // QKV projection
    // Producer warps: prefetch RoPE tables and norm weights for next phase
    // Consumer warps: do the actual matmul

    if (PRODUCER_WARPS > 0 && is_producer) {
        // Prefetch next phase data (QK norm weights and RoPE tables)
        int prefetch_start = (warp_id * WARP_SIZE + lane_id);
        ws_prefetch_range(q_norm_weight, prefetch_start, HEAD_DIM, PRODUCER_WARPS * WARP_SIZE);
        ws_prefetch_range(k_norm_weight, prefetch_start, HEAD_DIM, PRODUCER_WARPS * WARP_SIZE);
        ws_prefetch_range(cos_table + position * HEAD_DIM, prefetch_start, HEAD_DIM, PRODUCER_WARPS * WARP_SIZE);
        ws_prefetch_range(sin_table + position * HEAD_DIM, prefetch_start, HEAD_DIM, PRODUCER_WARPS * WARP_SIZE);
    } else if (!is_producer) {
        // Consumer warps do matmul
        constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
        int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
        int row_start = block_id * rows_per_block;
        int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

        for (int m_base = row_start; m_base < row_end; m_base += active_warps) {
            int m = m_base + consumer_warp_id;

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

                sum = ws_warp_reduce_sum(sum);
                if (lane_id == 0) {
                    *output_ptr = sum;
                }
            }
        }
    }

    grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache (no warp spec - already fast)
// =============================================================================

__device__ void ws_qk_norm_rope_cache(
    cg::grid_group& grid,
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
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    // Process Q heads
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }
        sum_sq = ws_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
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

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }
        sum_sq = ws_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
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

    grid.sync();
}

// =============================================================================
// Attention (keep block divergence from main kernel)
// =============================================================================

__device__ void ws_prefetch_weights_l2(
    const __nv_bfloat16* __restrict__ weights,
    int num_elements
) {
    float dummy = 0.0f;
    for (int i = threadIdx.x; i < num_elements; i += LDG_BLOCK_SIZE * 4) {
        dummy += __bfloat162float(__ldg(weights + i));
    }
    __shared__ float s_dummy;
    if (threadIdx.x == 0) s_dummy = dummy;
}

__device__ void ws_attention(
    cg::grid_group& grid,
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
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int ATTN_BLOCKS = NUM_Q_HEADS;

    if (block_id >= ATTN_BLOCKS) {
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;

        if (prefetch_block_id < num_prefetch_blocks / 3) {
            int elems_per_block = (Q_SIZE * HIDDEN_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = prefetch_block_id * elems_per_block;
            ws_prefetch_weights_l2(o_weight + start, elems_per_block);
        }
        else if (prefetch_block_id < 2 * num_prefetch_blocks / 3) {
            int adjusted_id = prefetch_block_id - num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            ws_prefetch_weights_l2(gate_weight + start, elems_per_block);
        }
        else {
            int adjusted_id = prefetch_block_id - 2 * num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            ws_prefetch_weights_l2(up_weight + start, elems_per_block);
        }

        grid.sync();
        return;
    }

    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

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

        for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
            }
            score = ws_warp_reduce_sum(score) * attn_scale;
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
            for (int w = 1; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < LDG_NUM_WARPS; w++) {
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

    grid.sync();
}

// =============================================================================
// O Projection + RMSNorm + MLP with warp specialization
// =============================================================================

__device__ void ws_o_proj_postnorm_mlp(
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
    __nv_bfloat16* __restrict__ hidden_out,
    // Next layer weights for prefetch
    const __nv_bfloat16* __restrict__ next_norm_weight,
    const __nv_bfloat16* __restrict__ next_q_weight,
    bool is_last_layer
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    bool is_producer = (PRODUCER_WARPS > 0) && (warp_id < PRODUCER_WARPS);
    int consumer_warp_id = warp_id - PRODUCER_WARPS;
    int active_warps = (PRODUCER_WARPS > 0) ? CONSUMER_WARPS : LDG_NUM_WARPS;

    // O Projection + Residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    if (PRODUCER_WARPS > 0 && is_producer) {
        // Prefetch post-attention norm weights
        int prefetch_start = (warp_id * WARP_SIZE + lane_id);
        ws_prefetch_range(post_norm_weight, prefetch_start, HIDDEN_SIZE, PRODUCER_WARPS * WARP_SIZE);
    } else if (!is_producer) {
        for (int m_base = hid_start; m_base < hid_end; m_base += active_warps) {
            int m = m_base + consumer_warp_id;

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

                sum = ws_warp_reduce_sum(sum);
                if (lane_id == 0) {
                    g_activations[m] = sum + g_residual[m];
                }
            }
        }
    }

    grid.sync();

    // Post-attention RMSNorm (block 0 only, all warps)
    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = g_activations[i];
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = ws_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ws_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            g_activations[i] = g_residual[i] * rstd * w;
        }
    }

    grid.sync();

    // Gate + Up + SiLU
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    if (PRODUCER_WARPS > 0 && is_producer) {
        // Prefetch down projection weights
        int prefetch_start = (warp_id * WARP_SIZE + lane_id);
        int total_down = HIDDEN_SIZE * INTERMEDIATE_SIZE;
        ws_prefetch_range(down_weight, prefetch_start, total_down / (PRODUCER_WARPS * 4), PRODUCER_WARPS * WARP_SIZE);
    } else if (!is_producer) {
        for (int m_base = int_start; m_base < int_end; m_base += active_warps) {
            int m = m_base + consumer_warp_id;

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

                gate_sum = ws_warp_reduce_sum(gate_sum);
                up_sum = ws_warp_reduce_sum(up_sum);

                if (lane_id == 0) {
                    g_mlp_intermediate[m] = ws_silu(gate_sum) * up_sum;
                }
            }
        }
    }

    grid.sync();

    // Down projection + residual
    if (PRODUCER_WARPS > 0 && is_producer && !is_last_layer && next_norm_weight != nullptr) {
        // Prefetch next layer's input layernorm and Q weights
        int prefetch_start = (warp_id * WARP_SIZE + lane_id);
        ws_prefetch_range(next_norm_weight, prefetch_start, HIDDEN_SIZE, PRODUCER_WARPS * WARP_SIZE);
        // Q weight is large, just prefetch first portion
        ws_prefetch_range(next_q_weight, prefetch_start, Q_SIZE * HIDDEN_SIZE / 16, PRODUCER_WARPS * WARP_SIZE);
    } else if (!is_producer) {
        for (int m_base = hid_start; m_base < hid_end; m_base += active_warps) {
            int m = m_base + consumer_warp_id;

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

                sum = ws_warp_reduce_sum(sum);
                if (lane_id == 0) {
                    float out_val = sum + g_residual[m];
                    g_activations[m] = out_val;
                    hidden_out[m] = __float2bfloat16(out_val);
                }
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Final Norm + LM Head
// =============================================================================

__device__ void ws_final_norm(
    cg::grid_group& grid,
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    float* __restrict__ g_normalized
) {
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = input[i];
            local_sum_sq += v * v;
        }

        local_sum_sq = ws_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ws_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            g_normalized[i] = input[i] * rstd * __bfloat162float(__ldg(norm_weight + i));
        }
    }

    grid.sync();
}

__global__ void ws_lm_head_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (LDG_VOCAB_SIZE + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    __shared__ float s_max_vals[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int s_max_idxs[LDG_LM_BLOCK_SIZE / WARP_SIZE];

    float thread_max = -INFINITY;
    int thread_max_idx = -1;

    for (int row_base = row_start; row_base < row_end; row_base += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
        int row = row_base + warp_id;

        if (row < row_end) {
            const __nv_bfloat16* weight_row = lm_head_weight + row * HIDDEN_SIZE;

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * input[k] +
                       __bfloat162float(w_ptr[1]) * input[k+1] +
                       __bfloat162float(w_ptr[2]) * input[k+2] +
                       __bfloat162float(w_ptr[3]) * input[k+3];
            }

            sum = ws_warp_reduce_sum(sum);

            if (lane_id == 0 && sum > thread_max) {
                thread_max = sum;
                thread_max_idx = row;
            }
        }
    }

    if (lane_id == 0) {
        s_max_vals[warp_id] = thread_max;
        s_max_idxs[warp_id] = thread_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_max = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? s_max_vals[lane_id] : -INFINITY;
        int warp_max_idx = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? s_max_idxs[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, warp_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, warp_max_idx, offset);
            if (other_max > warp_max) {
                warp_max = other_max;
                warp_max_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[block_id] = warp_max;
            block_max_idxs[block_id] = warp_max_idx;
        }
    }
}

__global__ void ws_reduce_max_kernel(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
    int lane_id = threadIdx.x % WARP_SIZE;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float val = block_max_vals[i];
        if (val > max_val) {
            max_val = val;
            max_idx = block_max_idxs[i];
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    if (lane_id == 0) {
        atomicMax(reinterpret_cast<int*>(output_token), max_idx);
    }
}

// =============================================================================
// Main Decode Kernel
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
ws_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const WarpSpecLayerWeights* __restrict__ layer_weights,
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

    const __nv_bfloat16* input = embed_weight + input_token_id * HIDDEN_SIZE;

    for (int layer = 0; layer < num_layers; layer++) {
        const WarpSpecLayerWeights& lw = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * NUM_KV_HEADS * max_seq_len * HEAD_DIM;
        __nv_bfloat16* layer_v_cache = v_cache + layer * NUM_KV_HEADS * max_seq_len * HEAD_DIM;

        // Get next layer weights for prefetch (if not last layer)
        const __nv_bfloat16* next_norm_weight = (layer < num_layers - 1) ?
            layer_weights[layer + 1].input_layernorm_weight : nullptr;
        const __nv_bfloat16* next_q_weight = (layer < num_layers - 1) ?
            layer_weights[layer + 1].q_proj_weight : nullptr;

        ws_matvec_qkv(grid, input, lw.input_layernorm_weight,
                      lw.q_proj_weight, lw.k_proj_weight, lw.v_proj_weight,
                      g_normalized, g_residual, g_q, g_k, g_v,
                      lw.q_norm_weight, lw.k_norm_weight, cos_table, sin_table, position);

        ws_qk_norm_rope_cache(grid, g_q, g_k, g_v,
                              lw.q_norm_weight, lw.k_norm_weight,
                              cos_table, sin_table,
                              layer_k_cache, layer_v_cache,
                              position, max_seq_len);

        ws_attention(grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
                     cache_len, max_seq_len, attn_scale,
                     lw.o_proj_weight, lw.gate_proj_weight, lw.up_proj_weight);

        ws_o_proj_postnorm_mlp(grid, lw.o_proj_weight, lw.post_attn_layernorm_weight,
                               lw.gate_proj_weight, lw.up_proj_weight, lw.down_proj_weight,
                               g_attn_out, g_residual, g_activations, g_mlp_intermediate,
                               hidden_buffer, next_norm_weight, next_q_weight,
                               layer == num_layers - 1);

        input = hidden_buffer;
    }

    ws_final_norm(grid, g_activations, final_norm_weight, g_normalized);
}

// =============================================================================
// Launch Function
// =============================================================================

extern "C" void launch_ws_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const WarpSpecLayerWeights* layer_weights,
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
        (void*)&embed_weight, (void*)&layer_weights, (void*)&final_norm_weight,
        (void*)&cos_table, (void*)&sin_table, (void*)&k_cache, (void*)&v_cache, (void*)&hidden_buffer,
        (void*)&g_activations, (void*)&g_residual, (void*)&g_q, (void*)&g_k, (void*)&g_v, (void*)&g_attn_out,
        (void*)&g_mlp_intermediate, (void*)&g_normalized,
        (void*)&num_layers, (void*)&position, (void*)&cache_len, (void*)&max_seq_len, (void*)&attn_scale
    };

    cudaLaunchCooperativeKernel(
        (void*)ws_decode_kernel,
        dim3(LDG_NUM_BLOCKS),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    // LM head phase 1: compute logits and find per-block max
    ws_lm_head_kernel<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    // LM head phase 2: reduce across blocks to find global argmax
    ws_reduce_max_kernel<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS
    );
}
