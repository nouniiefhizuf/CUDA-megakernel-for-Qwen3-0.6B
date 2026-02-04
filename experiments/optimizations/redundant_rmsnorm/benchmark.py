#!/usr/bin/env python3
"""
Benchmark and verify Redundant RMSNorm optimization.

This optimization eliminates 2 grid.sync() calls per layer by having ALL thread blocks
compute RMSNorm redundantly instead of only block 0.

Expected improvement:
- 28 layers * 2 RMSNorm phases = 56 grid.sync() calls eliminated
- grid.sync() overhead is ~5-10us each on cooperative kernel
- Expected savings: 280-560us per token

Usage:
    python benchmark.py                    # Full benchmark
    python benchmark.py --correctness      # Correctness check only
    python benchmark.py --warmup 50 --iters 200  # Custom iterations
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

# Model configuration
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 3072
MAX_SEQ_LEN = 512


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def get_project_paths():
    """Get absolute paths for kernel directories."""
    # Find project root by looking for csrc directory
    current = os.path.dirname(os.path.abspath(__file__))
    while current != '/':
        if os.path.exists(os.path.join(current, "csrc", "megakernel", "config.cuh")):
            return current, os.path.join(current, "csrc", "megakernel")
        current = os.path.dirname(current)
    raise RuntimeError("Could not find project root (looking for csrc/megakernel/config.cuh)")


def compile_original_kernel():
    """Compile the original kernel for comparison."""
    project_root, kernel_dir = get_project_paths()

    with open(os.path.join(kernel_dir, "fused_decode_ldg.cu")) as f:
        cuda_src = f.read()

    cpp_src = get_cpp_bindings("Original", "LDGLayerWeights", "launch_ldg_decode",
                               "init_ldg_layer_weights", "init_ldg_embed_weight", "decode_ldg")

    module = load_inline(
        name="original_kernel_benchmark",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
            "-lineinfo",
            "-maxrregcount=64",
        ],
        verbose=False,
    )

    return module


def compile_optimized_kernel():
    """Compile the redundant RMSNorm optimized kernel."""
    project_root, kernel_dir = get_project_paths()
    opt_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(opt_dir, "kernel.cu")) as f:
        cuda_src = f.read()

    cpp_src = get_cpp_bindings("Optimized", "RRMSNormLayerWeights", "launch_rrmsnorm_decode",
                               "init_rrmsnorm_layer_weights", "init_rrmsnorm_embed_weight", "decode_rrmsnorm")

    module = load_inline(
        name="rrmsnorm_kernel_benchmark",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
            "-lineinfo",
            "-maxrregcount=64",
        ],
        verbose=False,
    )

    return module


def get_cpp_bindings(name: str, weights_struct: str, launch_fn: str, init_weights_fn: str, init_embed_fn: str, decode_fn: str):
    """Generate C++ bindings for a kernel."""
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

struct {weights_struct} {{
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
}};

extern "C" void {launch_fn}(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const {weights_struct}* layer_weights,
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
);

static std::vector<{weights_struct}> g_{name}_layer_weights;
static {weights_struct}* d_{name}_layer_weights = nullptr;
static torch::Tensor d_{name}_embed_weight;

void {init_weights_fn}(
    std::vector<torch::Tensor> input_layernorm_weights,
    std::vector<torch::Tensor> q_proj_weights,
    std::vector<torch::Tensor> k_proj_weights,
    std::vector<torch::Tensor> v_proj_weights,
    std::vector<torch::Tensor> q_norm_weights,
    std::vector<torch::Tensor> k_norm_weights,
    std::vector<torch::Tensor> o_proj_weights,
    std::vector<torch::Tensor> post_attn_layernorm_weights,
    std::vector<torch::Tensor> gate_proj_weights,
    std::vector<torch::Tensor> up_proj_weights,
    std::vector<torch::Tensor> down_proj_weights
) {{
    int num_layers = input_layernorm_weights.size();
    g_{name}_layer_weights.resize(num_layers);

    for (int i = 0; i < num_layers; i++) {{
        g_{name}_layer_weights[i].input_layernorm_weight = input_layernorm_weights[i].data_ptr();
        g_{name}_layer_weights[i].q_proj_weight = q_proj_weights[i].data_ptr();
        g_{name}_layer_weights[i].k_proj_weight = k_proj_weights[i].data_ptr();
        g_{name}_layer_weights[i].v_proj_weight = v_proj_weights[i].data_ptr();
        g_{name}_layer_weights[i].q_norm_weight = q_norm_weights[i].data_ptr();
        g_{name}_layer_weights[i].k_norm_weight = k_norm_weights[i].data_ptr();
        g_{name}_layer_weights[i].o_proj_weight = o_proj_weights[i].data_ptr();
        g_{name}_layer_weights[i].post_attn_layernorm_weight = post_attn_layernorm_weights[i].data_ptr();
        g_{name}_layer_weights[i].gate_proj_weight = gate_proj_weights[i].data_ptr();
        g_{name}_layer_weights[i].up_proj_weight = up_proj_weights[i].data_ptr();
        g_{name}_layer_weights[i].down_proj_weight = down_proj_weights[i].data_ptr();
    }}

    if (d_{name}_layer_weights != nullptr) {{
        cudaFree(d_{name}_layer_weights);
    }}
    cudaMalloc(&d_{name}_layer_weights, num_layers * sizeof({weights_struct}));
    cudaMemcpy(d_{name}_layer_weights, g_{name}_layer_weights.data(), num_layers * sizeof({weights_struct}), cudaMemcpyHostToDevice);
}}

void {init_embed_fn}(torch::Tensor embed_weight) {{
    d_{name}_embed_weight = embed_weight;
}}

int {decode_fn}(
    int input_token_id,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden_buffer,
    torch::Tensor g_activations,
    torch::Tensor g_residual,
    torch::Tensor g_q,
    torch::Tensor g_k,
    torch::Tensor g_v,
    torch::Tensor g_attn_out,
    torch::Tensor g_mlp_intermediate,
    torch::Tensor g_normalized,
    torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len
) {{
    float attn_scale = 1.0f / sqrtf(128.0f);
    auto output_token = torch::empty({{1}}, torch::dtype(torch::kInt32).device(k_cache.device()));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    {launch_fn}(
        input_token_id,
        output_token.data_ptr<int>(),
        d_{name}_embed_weight.data_ptr(),
        d_{name}_layer_weights,
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden_buffer.data_ptr(),
        g_activations.data_ptr(),
        g_residual.data_ptr(),
        g_q.data_ptr(),
        g_k.data_ptr(),
        g_v.data_ptr(),
        g_attn_out.data_ptr(),
        g_mlp_intermediate.data_ptr(),
        g_normalized.data_ptr(),
        block_max_vals.data_ptr(),
        block_max_idxs.data_ptr(),
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        stream
    );

    cudaStreamSynchronize(stream);
    return output_token.item<int>();
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{init_weights_fn}", &{init_weights_fn});
    m.def("{init_embed_fn}", &{init_embed_fn});
    m.def("{decode_fn}", &{decode_fn});
}}
"""


def load_weights_from_hf():
    """Load weights from HuggingFace."""
    from transformers import AutoModelForCausalLM
    print("Loading model weights from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return state_dict


class KernelWrapper:
    """Wrapper to manage kernel buffers and weights."""

    def __init__(self, kernel, init_weights_fn, init_embed_fn, decode_fn, state_dict):
        self.kernel = kernel
        self.decode_fn = decode_fn
        self.device = "cuda"

        # Extract weights
        input_layernorm_weights = []
        q_proj_weights = []
        k_proj_weights = []
        v_proj_weights = []
        q_norm_weights = []
        k_norm_weights = []
        o_proj_weights = []
        post_attn_layernorm_weights = []
        gate_proj_weights = []
        up_proj_weights = []
        down_proj_weights = []

        for i in range(NUM_LAYERS):
            input_layernorm_weights.append(state_dict[f"model.layers.{i}.input_layernorm.weight"].contiguous())
            q_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"].contiguous())
            k_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"].contiguous())
            v_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"].contiguous())
            q_norm_weights.append(state_dict[f"model.layers.{i}.self_attn.q_norm.weight"].contiguous())
            k_norm_weights.append(state_dict[f"model.layers.{i}.self_attn.k_norm.weight"].contiguous())
            o_proj_weights.append(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"].contiguous())
            post_attn_layernorm_weights.append(state_dict[f"model.layers.{i}.post_attention_layernorm.weight"].contiguous())
            gate_proj_weights.append(state_dict[f"model.layers.{i}.mlp.gate_proj.weight"].contiguous())
            up_proj_weights.append(state_dict[f"model.layers.{i}.mlp.up_proj.weight"].contiguous())
            down_proj_weights.append(state_dict[f"model.layers.{i}.mlp.down_proj.weight"].contiguous())

        self.final_norm_weight = state_dict["model.norm.weight"].contiguous()
        self.lm_head_weight = state_dict["lm_head.weight"].contiguous()
        embed_weight = state_dict["model.embed_tokens.weight"].contiguous()

        self.cos_table, self.sin_table = precompute_rope_freqs(HEAD_DIM, MAX_SEQ_LEN, 1000000.0, self.device)

        # Initialize kernel weights
        getattr(kernel, init_weights_fn)(
            input_layernorm_weights, q_proj_weights, k_proj_weights, v_proj_weights,
            q_norm_weights, k_norm_weights, o_proj_weights, post_attn_layernorm_weights,
            gate_proj_weights, up_proj_weights, down_proj_weights,
        )
        getattr(kernel, init_embed_fn)(embed_weight)

        # Allocate buffers
        HIGHPAR_NUM_BLOCKS = 1184
        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        self.hidden_buffer = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.bfloat16)
        self.g_activations = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.g_residual = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.g_q = torch.zeros(16 * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_k = torch.zeros(8 * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_v = torch.zeros(8 * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_attn_out = torch.zeros(16 * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_mlp_intermediate = torch.zeros(INTERMEDIATE_SIZE, device=self.device, dtype=torch.float32)
        self.g_normalized = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.block_max_vals = torch.zeros(HIGHPAR_NUM_BLOCKS, device=self.device, dtype=torch.float32)
        self.block_max_idxs = torch.zeros(HIGHPAR_NUM_BLOCKS, device=self.device, dtype=torch.int32)

    def reset_cache(self):
        """Reset KV cache for fresh generation."""
        self.k_cache.zero_()
        self.v_cache.zero_()

    def decode(self, token_id: int, position: int) -> int:
        """Run a single decode step."""
        return getattr(self.kernel, self.decode_fn)(
            token_id,
            self.final_norm_weight,
            self.lm_head_weight,
            self.cos_table,
            self.sin_table,
            self.k_cache,
            self.v_cache,
            self.hidden_buffer,
            self.g_activations,
            self.g_residual,
            self.g_q,
            self.g_k,
            self.g_v,
            self.g_attn_out,
            self.g_mlp_intermediate,
            self.g_normalized,
            self.block_max_vals,
            self.block_max_idxs,
            NUM_LAYERS,
            position,
            position + 1,  # cache_len
            MAX_SEQ_LEN,
        )


def verify_correctness(original: KernelWrapper, optimized: KernelWrapper, num_tokens: int = 10):
    """Verify that optimized kernel produces identical outputs."""
    print(f"\n{'='*60}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*60}")

    # Reset caches
    original.reset_cache()
    optimized.reset_cache()

    # Test with same input sequence
    test_tokens = [1234, 5678, 9012, 3456, 7890, 2345, 6789, 1357, 2468, 8024][:num_tokens]

    original_outputs = []
    optimized_outputs = []

    for i, token in enumerate(test_tokens):
        orig_out = original.decode(token, i)
        opt_out = optimized.decode(token, i)
        original_outputs.append(orig_out)
        optimized_outputs.append(opt_out)

    # Compare outputs
    all_match = True
    for i, (orig, opt) in enumerate(zip(original_outputs, optimized_outputs)):
        match = "PASS" if orig == opt else "FAIL"
        if orig != opt:
            all_match = False
        print(f"  Token {i}: Original={orig}, Optimized={opt} [{match}]")

    if all_match:
        print(f"\n  All {num_tokens} tokens match!")
    else:
        print(f"\n  MISMATCH DETECTED!")

    return all_match


def benchmark_kernel(wrapper: KernelWrapper, name: str, warmup: int = 50, iters: int = 200):
    """Benchmark a kernel's decode performance."""
    wrapper.reset_cache()

    # Warmup
    print(f"  Warming up {name}...")
    for i in range(warmup):
        wrapper.decode(1234, i % MAX_SEQ_LEN)

    torch.cuda.synchronize()
    wrapper.reset_cache()

    # Benchmark at different sequence positions
    positions = [1, 10, 50, 100, 200]
    results = {}

    for pos in positions:
        # Reset and fill cache up to position
        wrapper.reset_cache()
        for i in range(pos):
            wrapper.decode(1234, i)

        torch.cuda.synchronize()

        # Benchmark decode at this position
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            wrapper.decode(5678, pos)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        results[pos] = (avg_time, std_time)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Redundant RMSNorm optimization")
    parser.add_argument("--correctness", action="store_true", help="Run correctness check only")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()

    print("="*60)
    print("Redundant RMSNorm Optimization Benchmark")
    print("="*60)
    print("\nOptimization: All 82 blocks compute RMSNorm redundantly")
    print("Expected benefit: Eliminate 2 grid.sync() per layer = 56 syncs")
    print()

    # Load weights
    state_dict = load_weights_from_hf()

    # Compile kernels
    print("Compiling original kernel...")
    original_kernel = compile_original_kernel()
    original = KernelWrapper(
        original_kernel,
        "init_ldg_layer_weights",
        "init_ldg_embed_weight",
        "decode_ldg",
        state_dict
    )

    print("Compiling optimized kernel...")
    optimized_kernel = compile_optimized_kernel()
    optimized = KernelWrapper(
        optimized_kernel,
        "init_rrmsnorm_layer_weights",
        "init_rrmsnorm_embed_weight",
        "decode_rrmsnorm",
        state_dict
    )

    # Verify correctness
    correct = verify_correctness(original, optimized)

    if args.correctness:
        sys.exit(0 if correct else 1)

    if not correct:
        print("\nSkipping benchmark due to correctness failure!")
        sys.exit(1)

    # Benchmark
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")

    print("\nBenchmarking original kernel...")
    original_results = benchmark_kernel(original, "Original", args.warmup, args.iters)

    print("Benchmarking optimized kernel...")
    optimized_results = benchmark_kernel(optimized, "Optimized", args.warmup, args.iters)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS (time per decode in ms)")
    print(f"{'='*60}")
    print(f"{'Position':<10} {'Original':>12} {'Optimized':>12} {'Speedup':>10}")
    print("-" * 50)

    for pos in sorted(original_results.keys()):
        orig_avg, orig_std = original_results[pos]
        opt_avg, opt_std = optimized_results[pos]
        speedup = orig_avg / opt_avg
        print(f"{pos:<10} {orig_avg:>8.3f}ms    {opt_avg:>8.3f}ms    {speedup:>6.2f}x")

    # Summary
    avg_orig = sum(r[0] for r in original_results.values()) / len(original_results)
    avg_opt = sum(r[0] for r in optimized_results.values()) / len(optimized_results)
    avg_speedup = avg_orig / avg_opt

    print("-" * 50)
    print(f"{'Average':<10} {avg_orig:>8.3f}ms    {avg_opt:>8.3f}ms    {avg_speedup:>6.2f}x")

    # Estimate sync savings
    sync_savings_us = (avg_orig - avg_opt) * 1000
    syncs_eliminated = 56  # 28 layers * 2 RMSNorm phases
    us_per_sync = sync_savings_us / syncs_eliminated if sync_savings_us > 0 else 0

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    print(f"  Time saved per token: {sync_savings_us:.1f} us")
    print(f"  Syncs eliminated: {syncs_eliminated}")
    print(f"  Est. time per sync: {us_per_sync:.1f} us")

    tok_per_s_orig = 1000.0 / avg_orig
    tok_per_s_opt = 1000.0 / avg_opt
    print(f"\n  Original: {tok_per_s_orig:.1f} tok/s")
    print(f"  Optimized: {tok_per_s_opt:.1f} tok/s")
    print(f"  Improvement: +{tok_per_s_opt - tok_per_s_orig:.1f} tok/s ({(avg_speedup-1)*100:.1f}%)")


if __name__ == "__main__":
    main()
