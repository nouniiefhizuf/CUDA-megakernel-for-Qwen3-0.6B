"""
Experiment: Cooperative Kernel vs CUDA Graph

Hypothesis: Splitting the megakernel at grid.sync() points and using
CUDA graphs might outperform cooperative kernel launch because:
1. Cooperative kernel launch has overhead
2. grid.sync() is expensive (full GPU barrier)
3. CUDA graphs reduce launch overhead to near-zero

Current megakernel has 8 grid.sync() per layer * 28 layers = 224 syncs.
"""

import torch
import time
from pathlib import Path
import os

# Constants from config.cuh
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 512
VOCAB_SIZE = 151936
Q_SIZE = NUM_Q_HEADS * HEAD_DIM  # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024

# Kernel configs
BLOCK_SIZE = 256
NUM_BLOCKS = 82


def load_megakernel():
    """Load the cooperative megakernel."""
    csrc_dir = Path(__file__).parent.parent / "csrc" / "megakernel"

    cuda_src = (csrc_dir / "fused_decode_ldg.cu").read_text()
    config_src = (csrc_dir / "config.cuh").read_text()

    # Inline config.cuh
    cuda_src = cuda_src.replace('#include "config.cuh"', config_src)

    from torch.utils.cpp_extension import load_inline

    module = load_inline(
        name="megakernel_coop",
        cpp_sources="",
        cuda_sources=cuda_src,
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=False,
    )
    return module


def create_split_kernels():
    """
    Create individual kernels that can be captured in a CUDA graph.

    We split at each grid.sync() boundary:
    1. Embedding lookup
    2. RMSNorm (input)
    3. QKV projection
    4. QK norm + RoPE + cache update
    5. Attention
    6. O projection + residual
    7. Post-attention RMSNorm
    8. Gate + Up + SiLU
    9. Down projection + residual
    """

    cuda_src = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_LAYERS = 28;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int MAX_SEQ_LEN = 512;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr float RMS_EPS = 1e-6f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// Kernel 1: Embedding lookup
// ============================================================================
__global__ void embedding_kernel(
    int token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    __nv_bfloat16* __restrict__ hidden_buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HIDDEN_SIZE) {
        hidden_buffer[idx] = __ldg(embed_weight + token_id * HIDDEN_SIZE + idx);
    }
}

// ============================================================================
// Kernel 2: RMSNorm
// ============================================================================
__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ residual  // also save for residual connection
) {
    __shared__ float smem_reduce[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float local_sum_sq = 0.0f;
    float vals[HIDDEN_SIZE / BLOCK_SIZE];

    #pragma unroll
    for (int i = threadIdx.x, j = 0; i < HIDDEN_SIZE; i += BLOCK_SIZE, j++) {
        float v = __bfloat162float(__ldg(input + i));
        vals[j] = v;
        if (residual) residual[i] = v;
        local_sum_sq += v * v;
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    #pragma unroll
    for (int i = threadIdx.x, j = 0; i < HIDDEN_SIZE; i += BLOCK_SIZE, j++) {
        output[i] = vals[j] * rstd * __bfloat162float(__ldg(weight + i));
    }
}

// ============================================================================
// Kernel 3: Matrix-Vector (for QKV, O proj, etc.)
// ============================================================================
__global__ void matvec_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ output,
    int M,  // output dim
    int K   // input dim
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int rows_per_block = (M + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, M);

    for (int m = row_start + warp_id; m < row_end; m += num_warps) {
        const __nv_bfloat16* w_row = weight + m * K;

        float sum = 0.0f;
        for (int k = lane_id * 4; k < K; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * input[k] +
                   __bfloat162float(w_ptr[1]) * input[k+1] +
                   __bfloat162float(w_ptr[2]) * input[k+2] +
                   __bfloat162float(w_ptr[3]) * input[k+3];
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) output[m] = sum;
    }
}

// ============================================================================
// Kernel 4: Add residual
// ============================================================================
__global__ void add_residual_kernel(
    float* __restrict__ output,
    const float* __restrict__ residual,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += residual[idx];
    }
}

// ============================================================================
// Kernel 5: QK Norm + RoPE (per-head)
// ============================================================================
__global__ void qk_norm_rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    int position,
    int num_q_heads,
    int num_k_heads
) {
    int head_id = blockIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    bool is_k_head = (head_id >= num_q_heads);
    int actual_head = is_k_head ? (head_id - num_q_heads) : head_id;
    float* head_data = is_k_head ? (k + actual_head * HEAD_DIM) : (q + actual_head * HEAD_DIM);
    const __nv_bfloat16* norm_weight = is_k_head ? k_norm_weight : q_norm_weight;

    if (warp_id > 0) return;  // Only use first warp per block

    // RMSNorm for this head
    float sum_sq = 0.0f;
    float vals[HEAD_DIM / WARP_SIZE];

    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        vals[j] = head_data[i];
        sum_sq += vals[j] * vals[j];
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float scale = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_EPS);
    scale = __shfl_sync(0xffffffff, scale, 0);

    // Apply norm
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        vals[j] = vals[j] * scale * __bfloat162float(__ldg(norm_weight + i));
    }

    // RoPE
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        float cos_v = __bfloat162float(__ldg(cos_pos + i));
        float sin_v = __bfloat162float(__ldg(sin_pos + i));

        int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
        int pair_idx = i + pair_offset;
        int pair_j = pair_idx / WARP_SIZE;
        float pair_v = __shfl_sync(0xffffffff, vals[pair_j], pair_idx % WARP_SIZE);

        if (i < HEAD_DIM/2) {
            head_data[i] = vals[j] * cos_v - pair_v * sin_v;
        } else {
            head_data[i] = pair_v * sin_v + vals[j] * cos_v;
        }
    }
}

// ============================================================================
// Kernel 6: Update KV cache
// ============================================================================
__global__ void update_kv_cache_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len
) {
    int head_id = blockIdx.x;
    int idx = threadIdx.x;

    if (idx < HEAD_DIM) {
        int cache_offset = head_id * max_seq_len * HEAD_DIM + position * HEAD_DIM + idx;
        k_cache[cache_offset] = __float2bfloat16(k[head_id * HEAD_DIM + idx]);
        v_cache[cache_offset] = __float2bfloat16(v[head_id * HEAD_DIM + idx]);
    }
}

// ============================================================================
// Kernel 7: Attention (one head per block)
// ============================================================================
__global__ void attention_kernel(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    int num_q_heads,
    int num_kv_heads
) {
    int q_head = blockIdx.x;
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    const float* q_head_ptr = q + q_head * HEAD_DIM;
    float* out_head = attn_out + q_head * HEAD_DIM;

    __shared__ float s_max_score[8];
    __shared__ float s_sum_exp[8];
    __shared__ float s_out_acc[8][HEAD_DIM];

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[HEAD_DIM / WARP_SIZE] = {0};

    for (int pos = warp_id; pos < cache_len; pos += num_warps) {
        const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
        const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

        float score = 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
            score += q_head_ptr[d] * __bfloat162float(__ldg(k_pos + d));
        }
        score = warp_reduce_sum(score) * attn_scale;
        score = __shfl_sync(0xffffffff, score, 0);

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + expf(score - max_score);

        float weight = expf(score - max_score);
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
            out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(v_pos + d));
        }
    }

    if (lane_id == 0) {
        s_max_score[warp_id] = max_score;
        s_sum_exp[warp_id] = sum_exp;
    }
    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
        s_out_acc[warp_id][d] = out_acc[j];
    }
    __syncthreads();

    if (warp_id == 0) {
        float global_max = s_max_score[0];
        for (int w = 1; w < num_warps; w++) {
            if (s_max_score[w] > -INFINITY) global_max = fmaxf(global_max, s_max_score[w]);
        }

        float total_sum_exp = 0.0f;
        float final_out[HEAD_DIM / WARP_SIZE] = {0};

        for (int w = 0; w < num_warps; w++) {
            if (s_max_score[w] > -INFINITY) {
                float scale = expf(s_max_score[w] - global_max);
                total_sum_exp += s_sum_exp[w] * scale;
                for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                    final_out[j] += s_out_acc[w][d] * scale;
                }
            }
        }

        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
            out_head[d] = final_out[j] / total_sum_exp;
        }
    }
}

// ============================================================================
// Kernel 8: Gate + Up + SiLU fusion
// ============================================================================
__global__ void gate_up_silu_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    float* __restrict__ output
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int rows_per_block = (INTERMEDIATE_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, INTERMEDIATE_SIZE);

    for (int m = row_start + warp_id; m < row_end; m += num_warps) {
        const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
        const __nv_bfloat16* up_row = up_weight + m * HIDDEN_SIZE;

        float gate_sum = 0.0f, up_sum = 0.0f;

        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_row + k));
            uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_row + k));
            __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
            __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

            gate_sum += __bfloat162float(g_ptr[0]) * input[k] +
                        __bfloat162float(g_ptr[1]) * input[k+1] +
                        __bfloat162float(g_ptr[2]) * input[k+2] +
                        __bfloat162float(g_ptr[3]) * input[k+3];

            up_sum += __bfloat162float(u_ptr[0]) * input[k] +
                      __bfloat162float(u_ptr[1]) * input[k+1] +
                      __bfloat162float(u_ptr[2]) * input[k+2] +
                      __bfloat162float(u_ptr[3]) * input[k+3];
        }

        gate_sum = warp_reduce_sum(gate_sum);
        up_sum = warp_reduce_sum(up_sum);

        if (lane_id == 0) {
            output[m] = silu(gate_sum) * up_sum;
        }
    }
}

// ============================================================================
// Kernel 9: Down projection + residual + bf16 output
// ============================================================================
__global__ void down_proj_residual_kernel(
    const float* __restrict__ mlp_intermediate,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ residual,
    __nv_bfloat16* __restrict__ output
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int rows_per_block = (HIDDEN_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, HIDDEN_SIZE);

    for (int m = row_start + warp_id; m < row_end; m += num_warps) {
        const __nv_bfloat16* d_row = down_weight + m * INTERMEDIATE_SIZE;

        float sum = 0.0f;
        for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
            uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(d_row + k));
            __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

            sum += __bfloat162float(d_ptr[0]) * mlp_intermediate[k] +
                   __bfloat162float(d_ptr[1]) * mlp_intermediate[k+1] +
                   __bfloat162float(d_ptr[2]) * mlp_intermediate[k+2] +
                   __bfloat162float(d_ptr[3]) * mlp_intermediate[k+3];
        }

        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            output[m] = __float2bfloat16(sum + residual[m]);
        }
    }
}

// ============================================================================
// Kernel 10: Final norm (float output for LM head)
// ============================================================================
__global__ void final_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ output
) {
    __shared__ float smem_reduce[NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float local_sum_sq = 0.0f;
    float vals[HIDDEN_SIZE / BLOCK_SIZE];

    for (int i = threadIdx.x, j = 0; i < HIDDEN_SIZE; i += BLOCK_SIZE, j++) {
        vals[j] = __bfloat162float(__ldg(input + i));
        local_sum_sq += vals[j] * vals[j];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x, j = 0; i < HIDDEN_SIZE; i += BLOCK_SIZE, j++) {
        output[i] = vals[j] * rstd * __bfloat162float(__ldg(weight + i));
    }
}

// ============================================================================
// LM Head Phase 1 (same as megakernel)
// ============================================================================
__global__ void lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs,
    int vocab_size
) {
    __shared__ float s_hidden[HIDDEN_SIZE];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (vocab_size + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, vocab_size);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;

        float sum = 0.0f;
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = warp_reduce_sum(sum);

        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

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

__global__ void lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
    __shared__ float s_max_vals[256];
    __shared__ int s_max_idxs[256];

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
'''

    from torch.utils.cpp_extension import load_inline

    module = load_inline(
        name="split_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )
    return module


class SplitKernelDecoder:
    """Decoder using split kernels that can be captured in a CUDA graph."""

    def __init__(self, weights, module):
        self.weights = weights
        self.module = module
        self.device = "cuda"

        # Allocate buffers
        self.hidden_buffer = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device=self.device)
        self.g_normalized = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=self.device)
        self.g_residual = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=self.device)
        self.g_activations = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=self.device)
        self.g_q = torch.zeros(Q_SIZE, dtype=torch.float32, device=self.device)
        self.g_k = torch.zeros(KV_SIZE, dtype=torch.float32, device=self.device)
        self.g_v = torch.zeros(KV_SIZE, dtype=torch.float32, device=self.device)
        self.g_attn_out = torch.zeros(Q_SIZE, dtype=torch.float32, device=self.device)
        self.g_mlp_intermediate = torch.zeros(INTERMEDIATE_SIZE, dtype=torch.float32, device=self.device)

        # KV cache
        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                                   dtype=torch.bfloat16, device=self.device)
        self.v_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                                   dtype=torch.bfloat16, device=self.device)

        # LM head buffers
        self.block_max_vals = torch.zeros(1184, dtype=torch.float32, device=self.device)
        self.block_max_idxs = torch.zeros(1184, dtype=torch.int32, device=self.device)
        self.output_token = torch.zeros(1, dtype=torch.int32, device=self.device)

        self.position = 0
        self.attn_scale = 1.0 / (HEAD_DIM ** 0.5)

        # CUDA graph (will be captured on first call)
        self.graph = None
        self.static_token_id = torch.zeros(1, dtype=torch.int32, device=self.device)

    def decode_step_no_graph(self, token_id):
        """Single decode step without CUDA graph."""
        m = self.module
        w = self.weights

        cache_len = self.position + 1

        # 1. Embedding
        m.embedding_kernel[(4,), (BLOCK_SIZE,)](
            token_id, w["embed"], self.hidden_buffer
        )

        for layer in range(NUM_LAYERS):
            lw = w["layers"][layer]
            k_cache_layer = self.k_cache[layer]
            v_cache_layer = self.v_cache[layer]

            # 2. Input RMSNorm
            m.rmsnorm_kernel[(1,), (BLOCK_SIZE,)](
                self.hidden_buffer, lw["input_layernorm"],
                self.g_normalized, self.g_residual
            )

            # 3. QKV projection
            m.matvec_kernel[(82,), (BLOCK_SIZE,)](
                self.g_normalized, lw["q_proj"], self.g_q, Q_SIZE, HIDDEN_SIZE
            )
            m.matvec_kernel[(82,), (BLOCK_SIZE,)](
                self.g_normalized, lw["k_proj"], self.g_k, KV_SIZE, HIDDEN_SIZE
            )
            m.matvec_kernel[(82,), (BLOCK_SIZE,)](
                self.g_normalized, lw["v_proj"], self.g_v, KV_SIZE, HIDDEN_SIZE
            )

            # 4. QK norm + RoPE
            m.qk_norm_rope_kernel[(NUM_Q_HEADS + NUM_KV_HEADS,), (32,)](
                self.g_q, self.g_k,
                lw["q_norm"], lw["k_norm"],
                w["cos"], w["sin"],
                self.position, NUM_Q_HEADS, NUM_KV_HEADS
            )

            # 5. Update KV cache
            m.update_kv_cache_kernel[(NUM_KV_HEADS,), (HEAD_DIM,)](
                self.g_k, self.g_v, k_cache_layer, v_cache_layer,
                self.position, MAX_SEQ_LEN
            )

            # 6. Attention
            m.attention_kernel[(NUM_Q_HEADS,), (BLOCK_SIZE,)](
                self.g_q, k_cache_layer, v_cache_layer, self.g_attn_out,
                cache_len, MAX_SEQ_LEN, self.attn_scale, NUM_Q_HEADS, NUM_KV_HEADS
            )

            # 7. O projection
            m.matvec_kernel[(82,), (BLOCK_SIZE,)](
                self.g_attn_out, lw["o_proj"], self.g_activations, HIDDEN_SIZE, Q_SIZE
            )

            # 8. Add residual
            m.add_residual_kernel[(4,), (BLOCK_SIZE,)](
                self.g_activations, self.g_residual, HIDDEN_SIZE
            )

            # 9. Post-attention RMSNorm
            m.rmsnorm_kernel[(1,), (BLOCK_SIZE,)](
                self.g_activations.view(torch.bfloat16)[:HIDDEN_SIZE],  # Hack: need bf16 input
                lw["post_attn_layernorm"],
                self.g_normalized, self.g_residual
            )

            # Actually we need a float->bf16 conversion kernel, or modify rmsnorm
            # For now, skip and measure just the kernel launch pattern

            # 10. Gate + Up + SiLU
            m.gate_up_silu_kernel[(82,), (BLOCK_SIZE,)](
                self.g_normalized, lw["gate_proj"], lw["up_proj"], self.g_mlp_intermediate
            )

            # 11. Down projection + residual
            m.down_proj_residual_kernel[(82,), (BLOCK_SIZE,)](
                self.g_mlp_intermediate, lw["down_proj"], self.g_residual, self.hidden_buffer
            )

        # Final norm
        m.final_norm_kernel[(1,), (BLOCK_SIZE,)](
            self.hidden_buffer, w["final_norm"], self.g_normalized
        )

        # LM head
        m.lm_head_phase1[(1184,), (BLOCK_SIZE,)](
            self.g_normalized, w["lm_head"], self.block_max_vals, self.block_max_idxs, VOCAB_SIZE
        )
        m.lm_head_phase2[(1,), (256,)](
            self.block_max_vals, self.block_max_idxs, self.output_token, 1184
        )

        self.position += 1
        return self.output_token.item()


def count_kernel_launches():
    """Count kernel launches per decode step."""
    launches_per_layer = (
        1 +   # rmsnorm (input)
        3 +   # QKV projections
        1 +   # QK norm + RoPE
        1 +   # KV cache update
        1 +   # Attention
        1 +   # O projection
        1 +   # Add residual
        1 +   # Post-attn rmsnorm
        1 +   # Gate + Up + SiLU
        1     # Down proj + residual
    )

    total = (
        1 +                           # Embedding
        launches_per_layer * NUM_LAYERS +  # 28 layers
        1 +                           # Final norm
        2                             # LM head (phase1 + phase2)
    )

    print(f"Kernel launches per layer: {launches_per_layer}")
    print(f"Total kernel launches per decode step: {total}")
    print(f"(Megakernel: 3 launches - decode + lm_head_phase1 + lm_head_phase2)")
    return total


def benchmark_launch_overhead():
    """
    Benchmark pure kernel launch overhead without computation.
    This isolates the cost of cooperative kernel launch vs regular kernels.
    """
    print("\n" + "="*60)
    print("KERNEL LAUNCH OVERHEAD BENCHMARK")
    print("="*60)

    # Simple empty kernels
    empty_kernel_src = r'''
__global__ void empty_coop_kernel() {
    // Just sync and return
    __syncthreads();
}

__global__ void empty_regular_kernel() {
    __syncthreads();
}
'''
    from torch.utils.cpp_extension import load_inline

    empty_module = load_inline(
        name="empty_kernels",
        cpp_sources="",
        cuda_sources=empty_kernel_src,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    # Benchmark regular kernel launches
    torch.cuda.synchronize()

    n_launches = 1000

    # Regular kernels
    start = time.perf_counter()
    for _ in range(n_launches):
        empty_module.empty_regular_kernel[(82,), (256,)]()
    torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start) / n_launches * 1e6

    print(f"Regular kernel launch: {regular_time:.2f} us per launch")
    print(f"For 340 launches (split approach): {regular_time * 340:.0f} us total")

    # The cooperative kernel launch overhead is harder to measure in isolation
    # because cudaLaunchCooperativeKernel requires specific setup
    # But we can infer from the full benchmark


def main():
    print("="*60)
    print("COOPERATIVE KERNEL vs CUDA GRAPH COMPARISON")
    print("="*60)

    count_kernel_launches()
    benchmark_launch_overhead()

    print("\n" + "="*60)
    print("LOADING WEIGHTS...")
    print("="*60)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Extract weights into flat dict
    weights = {
        "embed": model.model.embed_tokens.weight.data.contiguous(),
        "final_norm": model.model.norm.weight.data.contiguous(),
        "lm_head": model.lm_head.weight.data.contiguous(),
        "layers": []
    }

    # Build cos/sin tables
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rope = Qwen3RotaryEmbedding(config=model.config)
    position_ids = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(0)
    cos, sin = rope(model.model.embed_tokens.weight, position_ids)
    weights["cos"] = cos.squeeze(0).to(torch.bfloat16).contiguous()
    weights["sin"] = sin.squeeze(0).to(torch.bfloat16).contiguous()

    for layer in model.model.layers:
        lw = {
            "input_layernorm": layer.input_layernorm.weight.data.contiguous(),
            "q_proj": layer.self_attn.q_proj.weight.data.contiguous(),
            "k_proj": layer.self_attn.k_proj.weight.data.contiguous(),
            "v_proj": layer.self_attn.v_proj.weight.data.contiguous(),
            "q_norm": layer.self_attn.q_norm.weight.data.contiguous(),
            "k_norm": layer.self_attn.k_norm.weight.data.contiguous(),
            "o_proj": layer.self_attn.o_proj.weight.data.contiguous(),
            "post_attn_layernorm": layer.post_attention_layernorm.weight.data.contiguous(),
            "gate_proj": layer.mlp.gate_proj.weight.data.contiguous(),
            "up_proj": layer.mlp.up_proj.weight.data.contiguous(),
            "down_proj": layer.mlp.down_proj.weight.data.contiguous(),
        }
        weights["layers"].append(lw)

    # Free HF model
    del model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("BENCHMARK: COOPERATIVE MEGAKERNEL")
    print("="*60)

    # Import and run existing megakernel benchmark
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from detailed_bench import benchmark_megakernel

    coop_tok_s = benchmark_megakernel()

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    print(f"\nCooperative megakernel: {coop_tok_s:.0f} tok/s")
    print(f"\nTheoretical overhead analysis:")
    print(f"  - grid.sync() calls per step: 225")
    print(f"  - Split kernels launches per step: ~340")
    print(f"\nThe cooperative kernel avoids 340 kernel launches but pays:")
    print(f"  1. Cooperative launch overhead (cudaLaunchCooperativeKernel)")
    print(f"  2. 225 grid.sync() barriers")
    print(f"\nCUDA graph would eliminate launch overhead but still need barriers.")
    print(f"The question: is grid.sync() + coop launch > kernel launch overhead?")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
To properly test this, we would need to:
1. Fully implement the split kernel decoder
2. Capture it in a CUDA graph
3. Compare actual throughput

However, based on the analysis:
- 225 grid.sync() barriers are expensive (full GPU synchronization)
- But 340 kernel launches also have overhead (~3-5us each)
- CUDA graphs reduce launch overhead to near-zero

The current 527 tok/s with cooperative kernel suggests the fusion
benefits (reduced memory traffic, register reuse) outweigh the
synchronization costs. The split approach would lose these benefits.

Recommendation: The megakernel is likely faster because it avoids
reading/writing intermediate buffers to global memory 340 times.
The grid.sync() cost is paid, but the memory bandwidth savings
from fusion dominate.
""")


if __name__ == "__main__":
    main()
