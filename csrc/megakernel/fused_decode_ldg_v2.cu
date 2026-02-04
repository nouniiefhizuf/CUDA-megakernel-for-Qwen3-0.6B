/**
 * Fused Decode with 128-bit vectorized loads (v2)
 *
 * Optimizations over v1:
 * 1. Use uint4 (128-bit) loads for weights instead of uint2 (64-bit)
 * 2. Use float4 for activation vector reads where possible
 * 3. Vectorize attention K/V cache loads with uint4
 * 4. Increase memory-level parallelism with more loads per thread
 */

#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Configuration
constexpr int LDG_V2_NUM_BLOCKS = 82;
constexpr int LDG_V2_BLOCK_SIZE = 256;
constexpr int LDG_V2_NUM_WARPS = LDG_V2_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_V2_RMS_EPS = 1e-6f;

// LM head
constexpr int LDG_V2_LM_NUM_BLOCKS = 1184;
constexpr int LDG_V2_LM_BLOCK_SIZE = 256;
constexpr int LDG_V2_VOCAB_SIZE = 151936;

struct LDGV2LayerWeights {
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

__device__ __forceinline__ float ldg_v2_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float ldg_v2_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Optimized matvec with 128-bit loads (uint4 = 8 bf16 elements)
// =============================================================================

__device__ void ldg_v2_matvec_qkv(
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
        __shared__ float smem_reduce[LDG_V2_NUM_WARPS];

        float local_sum_sq = 0.0f;

        // Use uint4 loads for input (8 bf16 at a time)
        // HIDDEN_SIZE=1024, 1024/8=128 uint4 loads total
        // 256 threads, each does 128/256 < 1, so we need stride-based loading
        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += LDG_V2_BLOCK_SIZE * 8) {
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

        local_sum_sq = ldg_v2_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_V2_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_v2_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_V2_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        // Vectorized output with uint4 loads for norm_weight
        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += LDG_V2_BLOCK_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(norm_weight + i));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

            // Store as float4 pairs for better coalescing
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

    // QKV projection with 128-bit (uint4) weight loads
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += LDG_V2_NUM_WARPS) {
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

            // Use uint4 loads (128-bit = 8 bf16 elements) for weights
            // Each thread processes 8 elements per iteration
            // lane_id * 8 gives each lane a different starting point
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                // 128-bit weight load
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

                // 128-bit activation loads (2 x float4)
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

            sum = ldg_v2_warp_reduce_sum(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache (vectorized)
// =============================================================================

__device__ void ldg_v2_qk_norm_rope_cache(
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

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_V2_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        // Use float4 loads for q_head (HEAD_DIM=128, 128/4=32 = WARP_SIZE)
        float sum_sq = 0.0f;
        float4 q_local4 = *reinterpret_cast<const float4*>(q_head + lane_id * 4);
        sum_sq += q_local4.x * q_local4.x + q_local4.y * q_local4.y +
                  q_local4.z * q_local4.z + q_local4.w * q_local4.w;

        sum_sq = ldg_v2_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_V2_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        // Load norm weight with uint2 (4 bf16 = 64 bits matches float4 for q)
        uint2 qn_u2 = __ldg(reinterpret_cast<const uint2*>(q_norm_weight + lane_id * 4));
        __nv_bfloat16* qn_ptr = reinterpret_cast<__nv_bfloat16*>(&qn_u2);

        // Apply norm
        float q_local[4];
        q_local[0] = q_local4.x * scale * __bfloat162float(qn_ptr[0]);
        q_local[1] = q_local4.y * scale * __bfloat162float(qn_ptr[1]);
        q_local[2] = q_local4.z * scale * __bfloat162float(qn_ptr[2]);
        q_local[3] = q_local4.w * scale * __bfloat162float(qn_ptr[3]);

        // Load cos/sin with uint2
        uint2 cos_u2 = __ldg(reinterpret_cast<const uint2*>(cos_pos + lane_id * 4));
        uint2 sin_u2 = __ldg(reinterpret_cast<const uint2*>(sin_pos + lane_id * 4));
        __nv_bfloat16* cos_ptr = reinterpret_cast<__nv_bfloat16*>(&cos_u2);
        __nv_bfloat16* sin_ptr = reinterpret_cast<__nv_bfloat16*>(&sin_u2);

        // RoPE: each lane handles 4 consecutive elements
        // For HEAD_DIM=128, half=64
        // lane 0-15 handle first half (indices 0-63), lane 16-31 handle second half (64-127)
        float4 q_out4;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = lane_id * 4 + j;
            float cos_v = __bfloat162float(cos_ptr[j]);
            float sin_v = __bfloat162float(sin_ptr[j]);

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_lane = pair_idx / 4;
            int pair_j = pair_idx % 4;

            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_lane);

            float result;
            if (i < HEAD_DIM/2) {
                result = q_local[j] * cos_v - pair_v * sin_v;
            } else {
                result = pair_v * sin_v + q_local[j] * cos_v;
            }

            // Store to q_out4
            if (j == 0) q_out4.x = result;
            else if (j == 1) q_out4.y = result;
            else if (j == 2) q_out4.z = result;
            else q_out4.w = result;
        }

        *reinterpret_cast<float4*>(q_head + lane_id * 4) = q_out4;
    }

    // Process K heads + cache (similar vectorization)
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_V2_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        // Use float4 loads
        float sum_sq = 0.0f;
        float4 k_local4 = *reinterpret_cast<const float4*>(k_head + lane_id * 4);
        float4 v_local4 = *reinterpret_cast<const float4*>(v_head + lane_id * 4);

        sum_sq += k_local4.x * k_local4.x + k_local4.y * k_local4.y +
                  k_local4.z * k_local4.z + k_local4.w * k_local4.w;

        sum_sq = ldg_v2_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_V2_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        // Load norm weight
        uint2 kn_u2 = __ldg(reinterpret_cast<const uint2*>(k_norm_weight + lane_id * 4));
        __nv_bfloat16* kn_ptr = reinterpret_cast<__nv_bfloat16*>(&kn_u2);

        float k_local[4];
        k_local[0] = k_local4.x * scale * __bfloat162float(kn_ptr[0]);
        k_local[1] = k_local4.y * scale * __bfloat162float(kn_ptr[1]);
        k_local[2] = k_local4.z * scale * __bfloat162float(kn_ptr[2]);
        k_local[3] = k_local4.w * scale * __bfloat162float(kn_ptr[3]);

        // Load cos/sin
        uint2 cos_u2 = __ldg(reinterpret_cast<const uint2*>(cos_pos + lane_id * 4));
        uint2 sin_u2 = __ldg(reinterpret_cast<const uint2*>(sin_pos + lane_id * 4));
        __nv_bfloat16* cos_ptr = reinterpret_cast<__nv_bfloat16*>(&cos_u2);
        __nv_bfloat16* sin_ptr = reinterpret_cast<__nv_bfloat16*>(&sin_u2);

        float4 k_out4;
        __nv_bfloat16 k_cache_local[4];
        __nv_bfloat16 v_cache_local[4];

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = lane_id * 4 + j;
            float cos_v = __bfloat162float(cos_ptr[j]);
            float sin_v = __bfloat162float(sin_ptr[j]);

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_lane = pair_idx / 4;
            int pair_j = pair_idx % 4;

            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_lane);

            float k_final;
            if (i < HEAD_DIM/2) {
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            } else {
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            }

            if (j == 0) k_out4.x = k_final;
            else if (j == 1) k_out4.y = k_final;
            else if (j == 2) k_out4.z = k_final;
            else k_out4.w = k_final;

            k_cache_local[j] = __float2bfloat16(k_final);

            // V cache
            float v_val;
            if (j == 0) v_val = v_local4.x;
            else if (j == 1) v_val = v_local4.y;
            else if (j == 2) v_val = v_local4.z;
            else v_val = v_local4.w;
            v_cache_local[j] = __float2bfloat16(v_val);
        }

        *reinterpret_cast<float4*>(k_head + lane_id * 4) = k_out4;

        // Store to cache as uint2 (4 bf16)
        uint2 k_cache_u2;
        uint2 v_cache_u2;
        memcpy(&k_cache_u2, k_cache_local, sizeof(uint2));
        memcpy(&v_cache_u2, v_cache_local, sizeof(uint2));
        *reinterpret_cast<uint2*>(k_cache_head + lane_id * 4) = k_cache_u2;
        *reinterpret_cast<uint2*>(v_cache_head + lane_id * 4) = v_cache_u2;
    }

    grid.sync();
}

// =============================================================================
// Attention with vectorized KV cache loads
// =============================================================================

__device__ void ldg_v2_attention(
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
        // Prefetch blocks (unchanged from v1)
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;

        float dummy = 0.0f;
        if (prefetch_block_id < num_prefetch_blocks / 3) {
            int elems_per_block = (Q_SIZE * HIDDEN_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = prefetch_block_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += LDG_V2_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(o_weight + start + i));
            }
        }
        else if (prefetch_block_id < 2 * num_prefetch_blocks / 3) {
            int adjusted_id = prefetch_block_id - num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += LDG_V2_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(gate_weight + start + i));
            }
        }
        else {
            int adjusted_id = prefetch_block_id - 2 * num_prefetch_blocks / 3;
            int elems_per_block = (HIDDEN_SIZE * INTERMEDIATE_SIZE) / (num_prefetch_blocks / 3 + 1);
            int start = adjusted_id * elems_per_block;
            for (int i = threadIdx.x; i < elems_per_block; i += LDG_V2_BLOCK_SIZE * 4) {
                dummy += __bfloat162float(__ldg(up_weight + start + i));
            }
        }
        __shared__ float s_dummy;
        if (threadIdx.x == 0) s_dummy = dummy;

        grid.sync();
        return;
    }

    __shared__ float s_max_score[LDG_V2_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_V2_NUM_WARPS];
    __shared__ float s_out_acc[LDG_V2_NUM_WARPS][HEAD_DIM];

    int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

    for (int qh = head_start; qh < head_end; qh++) {
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        const float* q_head = q + qh * HEAD_DIM;
        float* out_head = attn_out + qh * HEAD_DIM;

        // Load Q into registers with float4
        float4 q_local = *reinterpret_cast<const float4*>(q_head + lane_id * 4);

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int pos = warp_id; pos < cache_len; pos += LDG_V2_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            // Use uint2 for K cache (4 bf16 elements matches lane's q_local)
            uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + lane_id * 4));
            __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(&k_u2);

            float score = q_local.x * __bfloat162float(k_ptr[0]) +
                          q_local.y * __bfloat162float(k_ptr[1]) +
                          q_local.z * __bfloat162float(k_ptr[2]) +
                          q_local.w * __bfloat162float(k_ptr[3]);

            score = ldg_v2_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);

            float weight = expf(score - max_score);

            // Load V with uint2
            uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_pos + lane_id * 4));
            __nv_bfloat16* v_ptr = reinterpret_cast<__nv_bfloat16*>(&v_u2);

            out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
            out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
            out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
            out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
        }

        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }

        // Store partial results
        s_out_acc[warp_id][lane_id * 4 + 0] = out_acc[0];
        s_out_acc[warp_id][lane_id * 4 + 1] = out_acc[1];
        s_out_acc[warp_id][lane_id * 4 + 2] = out_acc[2];
        s_out_acc[warp_id][lane_id * 4 + 3] = out_acc[3];
        __syncthreads();

        if (warp_id == 0) {
            float global_max = s_max_score[0];
            for (int w = 1; w < LDG_V2_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < LDG_V2_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float scale_w = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * scale_w;

                    final_out[0] += s_out_acc[w][lane_id * 4 + 0] * scale_w;
                    final_out[1] += s_out_acc[w][lane_id * 4 + 1] * scale_w;
                    final_out[2] += s_out_acc[w][lane_id * 4 + 2] * scale_w;
                    final_out[3] += s_out_acc[w][lane_id * 4 + 3] * scale_w;
                }
            }

            float4 out4;
            out4.x = final_out[0] / total_sum_exp;
            out4.y = final_out[1] / total_sum_exp;
            out4.z = final_out[2] / total_sum_exp;
            out4.w = final_out[3] / total_sum_exp;
            *reinterpret_cast<float4*>(out_head + lane_id * 4) = out4;
        }
        __syncthreads();
    }

    grid.sync();
}

// =============================================================================
// O Projection + PostNorm + MLP with 128-bit loads
// =============================================================================

__device__ void ldg_v2_o_proj_postnorm_mlp(
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

    // O Projection + Residual with 128-bit loads
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_V2_NUM_WARPS) {
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

            sum = ldg_v2_warp_reduce_sum(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    grid.sync();

    // Post-attention RMSNorm with vectorization
    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_V2_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x * 4; i < HIDDEN_SIZE; i += LDG_V2_BLOCK_SIZE * 4) {
            float4 v4 = *reinterpret_cast<const float4*>(g_activations + i);
            *reinterpret_cast<float4*>(g_residual + i) = v4;
            local_sum_sq += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
        }

        local_sum_sq = ldg_v2_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_V2_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_v2_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_V2_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += LDG_V2_BLOCK_SIZE * 8) {
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

    // Gate + Up + SiLU with 128-bit loads
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    for (int m_base = int_start; m_base < int_end; m_base += LDG_V2_NUM_WARPS) {
        int m = m_base + warp_id;

        if (m < int_end) {
            const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
            const __nv_bfloat16* up_row = up_weight + m * HIDDEN_SIZE;

            float gate_sum = 0.0f, up_sum = 0.0f;

            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 g_u4 = __ldg(reinterpret_cast<const uint4*>(gate_row + k));
                uint4 u_u4 = __ldg(reinterpret_cast<const uint4*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u4);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u4);

                float4 a1 = *reinterpret_cast<const float4*>(g_activations + k);
                float4 a2 = *reinterpret_cast<const float4*>(g_activations + k + 4);

                gate_sum += __bfloat162float(g_ptr[0]) * a1.x +
                            __bfloat162float(g_ptr[1]) * a1.y +
                            __bfloat162float(g_ptr[2]) * a1.z +
                            __bfloat162float(g_ptr[3]) * a1.w +
                            __bfloat162float(g_ptr[4]) * a2.x +
                            __bfloat162float(g_ptr[5]) * a2.y +
                            __bfloat162float(g_ptr[6]) * a2.z +
                            __bfloat162float(g_ptr[7]) * a2.w;

                up_sum += __bfloat162float(u_ptr[0]) * a1.x +
                          __bfloat162float(u_ptr[1]) * a1.y +
                          __bfloat162float(u_ptr[2]) * a1.z +
                          __bfloat162float(u_ptr[3]) * a1.w +
                          __bfloat162float(u_ptr[4]) * a2.x +
                          __bfloat162float(u_ptr[5]) * a2.y +
                          __bfloat162float(u_ptr[6]) * a2.z +
                          __bfloat162float(u_ptr[7]) * a2.w;
            }

            gate_sum = ldg_v2_warp_reduce_sum(gate_sum);
            up_sum = ldg_v2_warp_reduce_sum(up_sum);

            if (lane_id == 0) {
                g_mlp_intermediate[m] = ldg_v2_silu(gate_sum) * up_sum;
            }
        }
    }

    grid.sync();

    // Down projection + residual with 128-bit loads
    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_V2_NUM_WARPS) {
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

            sum = ldg_v2_warp_reduce_sum(sum);
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

__global__ void __launch_bounds__(LDG_V2_BLOCK_SIZE, 1)
ldg_v2_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const LDGV2LayerWeights* __restrict__ layer_weights,
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

    // Embedding lookup with uint4 (128-bit) loads
    const __nv_bfloat16* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * LDG_V2_BLOCK_SIZE * 8 + threadIdx.x * 8; i < HIDDEN_SIZE; i += num_blocks * LDG_V2_BLOCK_SIZE * 8) {
        uint4 e_u4 = __ldg(reinterpret_cast<const uint4*>(embed_row + i));
        *reinterpret_cast<uint4*>(hidden_buffer + i) = e_u4;
    }
    grid.sync();

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LDGV2LayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

        ldg_v2_matvec_qkv(
            grid, hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v
        );

        ldg_v2_qk_norm_rope_cache(
            grid, g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len
        );

        ldg_v2_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale,
            w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight
        );

        ldg_v2_o_proj_postnorm_mlp(
            grid, w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer
        );
    }

    // Final RMSNorm with vectorization
    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_V2_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += LDG_V2_BLOCK_SIZE * 8) {
            uint4 h_u4 = *reinterpret_cast<const uint4*>(hidden_buffer + i);
            __nv_bfloat16* h_ptr = reinterpret_cast<__nv_bfloat16*>(&h_u4);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float v = __bfloat162float(h_ptr[j]);
                g_activations[i + j] = v;
                local_sum_sq += v * v;
            }
        }

        local_sum_sq = ldg_v2_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_V2_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_v2_warp_reduce_sum(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_V2_RMS_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x * 8; i < HIDDEN_SIZE; i += LDG_V2_BLOCK_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(final_norm_weight + i));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

            float4 out1, out2;
            out1.x = g_activations[i + 0] * rstd * __bfloat162float(w_ptr[0]);
            out1.y = g_activations[i + 1] * rstd * __bfloat162float(w_ptr[1]);
            out1.z = g_activations[i + 2] * rstd * __bfloat162float(w_ptr[2]);
            out1.w = g_activations[i + 3] * rstd * __bfloat162float(w_ptr[3]);
            out2.x = g_activations[i + 4] * rstd * __bfloat162float(w_ptr[4]);
            out2.y = g_activations[i + 5] * rstd * __bfloat162float(w_ptr[5]);
            out2.z = g_activations[i + 6] * rstd * __bfloat162float(w_ptr[6]);
            out2.w = g_activations[i + 7] * rstd * __bfloat162float(w_ptr[7]);

            *reinterpret_cast<float4*>(g_normalized + i) = out1;
            *reinterpret_cast<float4*>(g_normalized + i + 4) = out2;
        }
    }
}

// =============================================================================
// LM Head (with 128-bit loads)
// =============================================================================

__global__ void ldg_v2_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    // Load hidden with float4
    for (int i = threadIdx.x * 4; i < HIDDEN_SIZE; i += LDG_V2_LM_BLOCK_SIZE * 4) {
        float4 h4 = *reinterpret_cast<const float4*>(hidden + i);
        s_hidden[i + 0] = h4.x;
        s_hidden[i + 1] = h4.y;
        s_hidden[i + 2] = h4.z;
        s_hidden[i + 3] = h4.w;
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (LDG_V2_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_V2_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += LDG_V2_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;

        float sum = 0.0f;
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k + 0] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k + 3] +
                   __bfloat162float(w_ptr[4]) * s_hidden[k + 4] +
                   __bfloat162float(w_ptr[5]) * s_hidden[k + 5] +
                   __bfloat162float(w_ptr[6]) * s_hidden[k + 6] +
                   __bfloat162float(w_ptr[7]) * s_hidden[k + 7];
        }
        sum = ldg_v2_warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[LDG_V2_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[LDG_V2_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < LDG_V2_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < LDG_V2_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

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

__global__ void ldg_v2_lm_head_phase2(
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

extern "C" void launch_ldg_v2_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGV2LayerWeights* layer_weights,
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
        (void*)ldg_v2_decode_kernel,
        dim3(LDG_V2_NUM_BLOCKS),
        dim3(LDG_V2_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    ldg_v2_lm_head_phase1<<<LDG_V2_LM_NUM_BLOCKS, LDG_V2_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    ldg_v2_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_V2_LM_NUM_BLOCKS
    );
}
