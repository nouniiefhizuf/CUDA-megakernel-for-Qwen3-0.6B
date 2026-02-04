/**
 * Fused Decode V4: cp.async Double-Buffering for MLP
 *
 * Strategy: Use __pipeline_memcpy_async to prefetch weight tiles into shared
 * memory while computing. Focus on MLP gate/up/down projections.
 *
 * Key insight: For matvec operations, the bottleneck is loading weights.
 * By double-buffering weight rows into shared memory, we can overlap memory
 * access latency with computation.
 *
 * This version caches the activation vector in shared memory (high reuse)
 * and streams weight rows through shared memory with double-buffering.
 */

#include "config.cuh"
#include <cooperative_groups.h>
#include <cuda_pipeline.h>

namespace cg = cooperative_groups;

// Configuration
constexpr int V4_NUM_BLOCKS = 82;
constexpr int V4_BLOCK_SIZE = 256;
constexpr int V4_NUM_WARPS = V4_BLOCK_SIZE / WARP_SIZE;
constexpr float V4_RMS_EPS = 1e-6f;

// Double-buffering: 2 sets of weight rows (gate + up combined)
// Each buffer: NUM_WARPS rows x HIDDEN_SIZE elements
// = 8 x 1024 x 2 bytes = 16KB per buffer for one projection
// For gate+up together: 32KB per buffer, 64KB total - within shared mem limits
constexpr int V4_MLP_TILE_ROWS = V4_NUM_WARPS;  // 8 rows per tile
constexpr int V4_MLP_ROW_SIZE = HIDDEN_SIZE;     // 1024 elements per row

// LM head
constexpr int V4_LM_NUM_BLOCKS = 1184;
constexpr int V4_LM_BLOCK_SIZE = 256;
constexpr int V4_VOCAB_SIZE = 151936;

struct V4LayerWeights {
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

__device__ __forceinline__ float v4_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float v4_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Async copy a tile of weight rows to shared memory
__device__ __forceinline__ void v4_async_load_tile(
    __nv_bfloat16* __restrict__ smem_dst,
    const __nv_bfloat16* __restrict__ gmem_base,
    int row_start,
    int num_rows,
    int row_size,
    int max_rows
) {
    // Each thread loads 16 bytes (8 bf16) at a time
    constexpr int CHUNK_BYTES = 16;
    constexpr int CHUNK_ELEMS = 8;
    
    int chunks_per_row = row_size / CHUNK_ELEMS;
    int total_chunks = num_rows * chunks_per_row;
    
    for (int c = threadIdx.x; c < total_chunks; c += V4_BLOCK_SIZE) {
        int row_in_tile = c / chunks_per_row;
        int chunk_in_row = c % chunks_per_row;
        int global_row = row_start + row_in_tile;
        int col = chunk_in_row * CHUNK_ELEMS;
        
        if (global_row < max_rows) {
            const void* src = gmem_base + global_row * row_size + col;
            void* dst = smem_dst + row_in_tile * row_size + col;
            __pipeline_memcpy_async(dst, src, CHUNK_BYTES);
        }
    }
}

// =============================================================================
// QKV Projection (same as v2 - already optimized)
// =============================================================================

__device__ void v4_matvec_qkv(
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
    float* __restrict__ v_out
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Block 0 does RMSNorm with vectorized loads
    if (block_id == 0) {
        __shared__ float smem[HIDDEN_SIZE];
        __shared__ float smem_reduce[V4_NUM_WARPS];

        float local_sum_sq = 0.0f;

        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE * 8) {
            uint4 in_u4 = __ldg(reinterpret_cast<const uint4*>(input + i));
            __nv_bfloat16* in_ptr = reinterpret_cast<__nv_bfloat16*>(&in_u4);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float v = __bfloat162float(in_ptr[j]);
                smem[i + j] = v;
                g_residual[i + j] = v;
                local_sum_sq += v * v;
            }
        }

        local_sum_sq = v4_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < V4_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = v4_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + V4_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(norm_weight + i));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

            float4 out1, out2;
            out1.x = smem[i + 0] * rstd * __bfloat162float(w_ptr[0]);
            out1.y = smem[i + 1] * rstd * __bfloat162float(w_ptr[1]);
            out1.z = smem[i + 2] * rstd * __bfloat162float(w_ptr[2]);
            out1.w = smem[i + 3] * rstd * __bfloat162float(w_ptr[3]);
            out2.x = smem[i + 4] * rstd * __bfloat162float(w_ptr[4]);
            out2.y = smem[i + 5] * rstd * __bfloat162float(w_ptr[5]);
            out2.z = smem[i + 6] * rstd * __bfloat162float(w_ptr[6]);
            out2.w = smem[i + 7] * rstd * __bfloat162float(w_ptr[7]);

            *reinterpret_cast<float4*>(g_normalized + i) = out1;
            *reinterpret_cast<float4*>(g_normalized + i + 4) = out2;
        }
    }

    grid.sync();

    // QKV projection with 128-bit loads
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += V4_NUM_WARPS) {
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
            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

                float4 act1 = *reinterpret_cast<const float4*>(g_normalized + k);
                float4 act2 = *reinterpret_cast<const float4*>(g_normalized + k + 4);

                sum += __bfloat162float(w_ptr[0]) * act1.x +
                       __bfloat162float(w_ptr[1]) * act1.y +
                       __bfloat162float(w_ptr[2]) * act1.z +
                       __bfloat162float(w_ptr[3]) * act1.w +
                       __bfloat162float(w_ptr[4]) * act2.x +
                       __bfloat162float(w_ptr[5]) * act2.y +
                       __bfloat162float(w_ptr[6]) * act2.z +
                       __bfloat162float(w_ptr[7]) * act2.w;
            }

            sum = v4_warp_reduce_sum(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache (same as v2)
// =============================================================================

__device__ void v4_qk_norm_rope_cache(
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

    for (int h = q_head_start + warp_id; h < q_head_end; h += V4_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }
        sum_sq = v4_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + V4_RMS_EPS);
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

    for (int h = k_head_start + warp_id; h < k_head_end; h += V4_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }
        sum_sq = v4_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + V4_RMS_EPS);
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
// Attention (same as v2)
// =============================================================================

__device__ void v4_attention(
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
        // Prefetch blocks
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;

        float dummy = 0.0f;
        if (prefetch_block_id < num_prefetch_blocks / 3) {
            int elems_per_block = (Q_SIZE * HIDDEN_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = prefetch_block_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += V4_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(o_weight + start + i));
            }
        }
        else if (prefetch_block_id < 2 * num_prefetch_blocks / 3) {
            int adjusted_id = prefetch_block_id - num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += V4_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(gate_weight + start + i));
            }
        }
        else {
            int adjusted_id = prefetch_block_id - 2 * num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += V4_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(up_weight + start + i));
            }
        }
        __shared__ float s_dummy;
        if (threadIdx.x == 0) s_dummy = dummy;

        grid.sync();
        return;
    }

    __shared__ float s_max_score[V4_NUM_WARPS];
    __shared__ float s_sum_exp[V4_NUM_WARPS];
    __shared__ float s_out_acc[V4_NUM_WARPS][HEAD_DIM];

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

        for (int pos = warp_id; pos < cache_len; pos += V4_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
            }
            score = v4_warp_reduce_sum(score) * attn_scale;
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
            for (int w = 1; w < V4_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < V4_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float scale_w = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * scale_w;

                    #pragma unroll
                    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                        final_out[j] += s_out_acc[w][d] * scale_w;
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
// O Projection + PostNorm + MLP with cp.async Double-Buffering
// =============================================================================

__device__ void v4_o_proj_postnorm_mlp_pipelined(
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

    // Shared memory for activation caching and weight double-buffering
    __shared__ float s_activations[HIDDEN_SIZE];
    
    // Double-buffer for gate+up weight rows
    // Each buffer: NUM_WARPS rows x HIDDEN_SIZE elements = 8 x 1024 = 8192 bf16 = 16KB per buffer
    __shared__ __nv_bfloat16 s_gate[2][V4_MLP_TILE_ROWS * V4_MLP_ROW_SIZE];
    __shared__ __nv_bfloat16 s_up[2][V4_MLP_TILE_ROWS * V4_MLP_ROW_SIZE];

    // O Projection + Residual (use __ldg - weights have low reuse)
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += V4_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* o_row = o_weight + m * Q_SIZE;

            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

                float4 a1 = *reinterpret_cast<const float4*>(attn_out + k);
                float4 a2 = *reinterpret_cast<const float4*>(attn_out + k + 4);

                sum += __bfloat162float(w_ptr[0]) * a1.x +
                       __bfloat162float(w_ptr[1]) * a1.y +
                       __bfloat162float(w_ptr[2]) * a1.z +
                       __bfloat162float(w_ptr[3]) * a1.w +
                       __bfloat162float(w_ptr[4]) * a2.x +
                       __bfloat162float(w_ptr[5]) * a2.y +
                       __bfloat162float(w_ptr[6]) * a2.z +
                       __bfloat162float(w_ptr[7]) * a2.w;
            }

            sum = v4_warp_reduce_sum(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    grid.sync();

    // Post-attention RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[V4_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x * 4; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE * 4) {
            float4 v4 = *reinterpret_cast<const float4*>(g_activations + i);
            *reinterpret_cast<float4*>(g_residual + i) = v4;
            local_sum_sq += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
        }

        local_sum_sq = v4_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < V4_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = v4_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + V4_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(post_norm_weight + i));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

            float4 r1 = *reinterpret_cast<const float4*>(g_residual + i);
            float4 r2 = *reinterpret_cast<const float4*>(g_residual + i + 4);

            float4 out1, out2;
            out1.x = r1.x * rstd * __bfloat162float(w_ptr[0]);
            out1.y = r1.y * rstd * __bfloat162float(w_ptr[1]);
            out1.z = r1.z * rstd * __bfloat162float(w_ptr[2]);
            out1.w = r1.w * rstd * __bfloat162float(w_ptr[3]);
            out2.x = r2.x * rstd * __bfloat162float(w_ptr[4]);
            out2.y = r2.y * rstd * __bfloat162float(w_ptr[5]);
            out2.z = r2.z * rstd * __bfloat162float(w_ptr[6]);
            out2.w = r2.w * rstd * __bfloat162float(w_ptr[7]);

            *reinterpret_cast<float4*>(g_activations + i) = out1;
            *reinterpret_cast<float4*>(g_activations + i + 4) = out2;
        }
    }

    grid.sync();

    // =========================================================================
    // Gate + Up + SiLU with cp.async double-buffering
    // =========================================================================
    
    // Load activations to shared memory (high reuse - read by 3072 output rows)
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE) {
        s_activations[i] = g_activations[i];
    }
    __syncthreads();
    
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);
    
    int num_tiles = (int_end - int_start + V4_MLP_TILE_ROWS - 1) / V4_MLP_TILE_ROWS;
    
    // Start loading first tile
    if (num_tiles > 0) {
        int tile_start = int_start;
        int tile_rows = min(V4_MLP_TILE_ROWS, int_end - tile_start);
        
        v4_async_load_tile(s_gate[0], gate_weight, tile_start, tile_rows, 
                           HIDDEN_SIZE, INTERMEDIATE_SIZE);
        v4_async_load_tile(s_up[0], up_weight, tile_start, tile_rows,
                           HIDDEN_SIZE, INTERMEDIATE_SIZE);
        __pipeline_commit();
    }
    
    int cur_buf = 0;
    
    // Process tiles with double-buffering
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = int_start + tile * V4_MLP_TILE_ROWS;
        int tile_rows = min(V4_MLP_TILE_ROWS, int_end - tile_start);
        
        // Start loading next tile (if exists)
        int next_tile = tile + 1;
        if (next_tile < num_tiles) {
            int next_tile_start = int_start + next_tile * V4_MLP_TILE_ROWS;
            int next_tile_rows = min(V4_MLP_TILE_ROWS, int_end - next_tile_start);
            int next_buf = 1 - cur_buf;
            
            v4_async_load_tile(s_gate[next_buf], gate_weight, next_tile_start, 
                               next_tile_rows, HIDDEN_SIZE, INTERMEDIATE_SIZE);
            v4_async_load_tile(s_up[next_buf], up_weight, next_tile_start,
                               next_tile_rows, HIDDEN_SIZE, INTERMEDIATE_SIZE);
            __pipeline_commit();
        }
        
        // Wait for current tile to be ready
        __pipeline_wait_prior(1);
        __syncthreads();
        
        // Compute: each warp handles one row in the tile
        int local_row = warp_id;
        int global_row = tile_start + local_row;
        
        if (local_row < tile_rows && global_row < int_end) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            
            // Dot product using shared memory weights and activations
            #pragma unroll 4
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                float a0 = s_activations[k];
                float a1 = s_activations[k + 1];
                float a2 = s_activations[k + 2];
                float a3 = s_activations[k + 3];
                
                __nv_bfloat16* g_ptr = &s_gate[cur_buf][local_row * HIDDEN_SIZE + k];
                __nv_bfloat16* u_ptr = &s_up[cur_buf][local_row * HIDDEN_SIZE + k];
                
                gate_sum += __bfloat162float(g_ptr[0]) * a0 +
                            __bfloat162float(g_ptr[1]) * a1 +
                            __bfloat162float(g_ptr[2]) * a2 +
                            __bfloat162float(g_ptr[3]) * a3;
                            
                up_sum += __bfloat162float(u_ptr[0]) * a0 +
                          __bfloat162float(u_ptr[1]) * a1 +
                          __bfloat162float(u_ptr[2]) * a2 +
                          __bfloat162float(u_ptr[3]) * a3;
            }
            
            gate_sum = v4_warp_reduce_sum(gate_sum);
            up_sum = v4_warp_reduce_sum(up_sum);
            
            if (lane_id == 0) {
                g_mlp_intermediate[global_row] = v4_silu(gate_sum) * up_sum;
            }
        }
        
        __syncthreads();
        cur_buf = 1 - cur_buf;
    }
    
    __pipeline_wait_prior(0);
    grid.sync();

    // =========================================================================
    // Down projection + residual (use __ldg - activations in global memory
    // have low reuse per weight row in this block-distributed scheme)
    // =========================================================================
    
    for (int m_base = hid_start; m_base < hid_end; m_base += V4_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* down_row = down_weight + m * INTERMEDIATE_SIZE;

            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
                uint4 d_u4 = __ldg(reinterpret_cast<const uint4*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u4);

                float4 m1 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k);
                float4 m2 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k + 4);

                sum += __bfloat162float(d_ptr[0]) * m1.x +
                       __bfloat162float(d_ptr[1]) * m1.y +
                       __bfloat162float(d_ptr[2]) * m1.z +
                       __bfloat162float(d_ptr[3]) * m1.w +
                       __bfloat162float(d_ptr[4]) * m2.x +
                       __bfloat162float(d_ptr[5]) * m2.y +
                       __bfloat162float(d_ptr[6]) * m2.z +
                       __bfloat162float(d_ptr[7]) * m2.w;
            }

            sum = v4_warp_reduce_sum(sum);
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

__global__ void __launch_bounds__(V4_BLOCK_SIZE, 1)
v4_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const V4LayerWeights* __restrict__ layer_weights,
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
    for (int i = block_id * V4_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * V4_BLOCK_SIZE) {
        hidden_buffer[i] = __ldg(embed_row + i);
    }
    grid.sync();

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const V4LayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        v4_matvec_qkv(
            grid, hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v
        );

        v4_qk_norm_rope_cache(
            grid, g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len
        );

        v4_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale,
            w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight
        );

        v4_o_proj_postnorm_mlp_pipelined(
            grid, w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer
        );
    }

    // Final RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[V4_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            g_activations[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = v4_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < V4_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = v4_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + V4_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += V4_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
}

// =============================================================================
// LM Head
// =============================================================================

__global__ void v4_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += V4_LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (V4_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, V4_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += V4_LM_BLOCK_SIZE / WARP_SIZE) {
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
        sum = v4_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[V4_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[V4_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < V4_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < V4_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

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

__global__ void v4_lm_head_phase2(
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

extern "C" void launch_v4_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const V4LayerWeights* layer_weights,
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
        (void*)v4_decode_kernel,
        dim3(V4_NUM_BLOCKS),
        dim3(V4_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    v4_lm_head_phase1<<<V4_LM_NUM_BLOCKS, V4_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    v4_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        V4_LM_NUM_BLOCKS
    );
}
