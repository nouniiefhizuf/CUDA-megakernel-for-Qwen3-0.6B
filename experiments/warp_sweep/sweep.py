#!/usr/bin/env python3
"""
Warp producer/consumer ratio sweep benchmark.

Tests different ratios of producer warps (prefetching) vs consumer warps (computing)
within each thread block to find optimal configuration.

Configuration:
- 256 threads per block = 8 warps
- Sweep: 0:8, 1:7, 2:6, 3:5, 4:4 (producer:consumer)

Usage:
    python sweep.py                 # Full sweep
    python sweep.py --ratio 2       # Test specific ratio only
    python sweep.py --positions 1,50,100,200  # Custom positions
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.cpp_extension import load_inline

# Model configuration
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
NUM_KV_HEADS = 8
NUM_Q_HEADS = 16
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
    current = os.path.dirname(os.path.abspath(__file__))
    while current != '/':
        if os.path.exists(os.path.join(current, "csrc", "megakernel", "config.cuh")):
            return current, os.path.join(current, "csrc", "megakernel")
        current = os.path.dirname(current)
    raise RuntimeError("Could not find project root")


def get_cpp_bindings(name: str):
    """Generate C++ bindings for the warp-spec kernel."""
    return f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

struct WarpSpecLayerWeights {{
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
);

static std::vector<WarpSpecLayerWeights> g_{name}_layer_weights;
static WarpSpecLayerWeights* d_{name}_layer_weights = nullptr;
static torch::Tensor d_{name}_embed_weight;

void init_{name}_layer_weights(
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
    cudaMalloc(&d_{name}_layer_weights, num_layers * sizeof(WarpSpecLayerWeights));
    cudaMemcpy(d_{name}_layer_weights, g_{name}_layer_weights.data(), num_layers * sizeof(WarpSpecLayerWeights), cudaMemcpyHostToDevice);
}}

void init_{name}_embed_weight(torch::Tensor embed_weight) {{
    d_{name}_embed_weight = embed_weight;
}}

int decode_{name}(
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

    launch_ws_decode(
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
    m.def("init_{name}_layer_weights", &init_{name}_layer_weights);
    m.def("init_{name}_embed_weight", &init_{name}_embed_weight);
    m.def("decode_{name}", &decode_{name});
}}
"""


def compile_kernel_variant(num_producer_warps: int):
    """Compile kernel with specific producer warp configuration."""
    project_root, kernel_dir = get_project_paths()
    sweep_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the warp specialization kernel
    with open(os.path.join(sweep_dir, "kernel_warp_spec.cu")) as f:
        cuda_src = f.read()

    name = f"pw{num_producer_warps}"
    cpp_src = get_cpp_bindings(name)

    print(f"  Compiling variant with {num_producer_warps} producer warps...")

    module = load_inline(
        name=f"warp_sweep_{name}_{num_producer_warps}",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
            "-lineinfo",
            f"-DNUM_PRODUCER_WARPS={num_producer_warps}",
        ],
        verbose=False,
    )

    return module, name


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

    def __init__(self, kernel, name: str, state_dict):
        self.kernel = kernel
        self.name = name
        self.decode_fn = f"decode_{name}"
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
        getattr(kernel, f"init_{name}_layer_weights")(
            input_layernorm_weights, q_proj_weights, k_proj_weights, v_proj_weights,
            q_norm_weights, k_norm_weights, o_proj_weights, post_attn_layernorm_weights,
            gate_proj_weights, up_proj_weights, down_proj_weights,
        )
        getattr(kernel, f"init_{name}_embed_weight")(embed_weight)

        # Allocate buffers
        HIGHPAR_NUM_BLOCKS = 1184
        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        self.hidden_buffer = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.bfloat16)
        self.g_activations = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.g_residual = torch.zeros(HIDDEN_SIZE, device=self.device, dtype=torch.float32)
        self.g_q = torch.zeros(NUM_Q_HEADS * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_k = torch.zeros(NUM_KV_HEADS * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_v = torch.zeros(NUM_KV_HEADS * HEAD_DIM, device=self.device, dtype=torch.float32)
        self.g_attn_out = torch.zeros(NUM_Q_HEADS * HEAD_DIM, device=self.device, dtype=torch.float32)
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


def benchmark_kernel(wrapper: KernelWrapper, positions: List[int], warmup: int = 50, iters: int = 200) -> Dict[int, Tuple[float, float]]:
    """Benchmark a kernel's decode performance at various positions."""
    wrapper.reset_cache()

    # Warmup
    for i in range(warmup):
        wrapper.decode(1234, i % MAX_SEQ_LEN)

    torch.cuda.synchronize()
    wrapper.reset_cache()

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
    parser = argparse.ArgumentParser(description="Warp producer/consumer ratio sweep")
    parser.add_argument("--ratio", type=int, default=None, help="Test specific producer warp count only (0-4)")
    parser.add_argument("--positions", type=str, default="1,50,100,200", help="Comma-separated positions to test")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    args = parser.parse_args()

    positions = [int(p) for p in args.positions.split(",")]
    ratios = [args.ratio] if args.ratio is not None else [0, 1, 2, 3, 4]

    print("=" * 70)
    print("Warp Producer/Consumer Ratio Sweep")
    print("=" * 70)
    print("\n8 warps per block (256 threads)")
    print(f"Testing ratios: {', '.join(f'{r}:{8-r}' for r in ratios)} (producer:consumer)")
    print(f"Positions: {positions}")
    print()

    # Load weights once
    state_dict = load_weights_from_hf()

    # Compile and benchmark each variant
    all_results = {}

    print("\nCompiling and benchmarking kernels...")
    for num_producers in ratios:
        try:
            kernel, name = compile_kernel_variant(num_producers)
            wrapper = KernelWrapper(kernel, name, state_dict)

            print(f"\n  Benchmarking {num_producers}:{8-num_producers} (producer:consumer)...")
            results = benchmark_kernel(wrapper, positions, args.warmup, args.iters)
            all_results[num_producers] = results

            # Print intermediate results
            for pos in positions:
                avg_ms, _ = results[pos]
                tok_s = 1000.0 / avg_ms
                print(f"    Position {pos}: {tok_s:.1f} tok/s ({avg_ms:.3f} ms)")

            # Clean up
            del wrapper
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[num_producers] = None

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS (tok/s)")
    print("=" * 70)

    # Header
    header = f"{'Ratio':<12}"
    for pos in positions:
        header += f"{'Pos ' + str(pos):>12}"
    header += f"{'Average':>12}"
    print(header)
    print("-" * len(header))

    # Data rows
    for num_producers in ratios:
        results = all_results.get(num_producers)
        if results is None:
            print(f"{num_producers}:{8-num_producers:<8} {'ERROR':>12}" * (len(positions) + 1))
            continue

        row = f"{num_producers}:{8-num_producers:<8}"
        tok_s_list = []
        for pos in positions:
            avg_ms, _ = results[pos]
            tok_s = 1000.0 / avg_ms
            tok_s_list.append(tok_s)
            row += f"{tok_s:>12.1f}"
        avg_tok_s = sum(tok_s_list) / len(tok_s_list)
        row += f"{avg_tok_s:>12.1f}"
        print(row)

    # Find best configuration
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline = all_results.get(0)
    if baseline:
        baseline_avg = sum(1000.0 / baseline[p][0] for p in positions) / len(positions)
        print(f"\nBaseline (0:8): {baseline_avg:.1f} tok/s average")

        best_ratio = 0
        best_avg = baseline_avg

        for num_producers in ratios:
            if num_producers == 0 or all_results.get(num_producers) is None:
                continue
            results = all_results[num_producers]
            avg = sum(1000.0 / results[p][0] for p in positions) / len(positions)
            speedup = avg / baseline_avg
            print(f"{num_producers}:{8-num_producers}: {avg:.1f} tok/s ({speedup:.2f}x vs baseline)")
            if avg > best_avg:
                best_avg = avg
                best_ratio = num_producers

        print(f"\nBest configuration: {best_ratio}:{8-best_ratio} (producer:consumer)")
        if best_ratio != 0:
            print(f"Improvement: {best_avg - baseline_avg:.1f} tok/s (+{(best_avg/baseline_avg - 1)*100:.1f}%)")
        else:
            print("No improvement from warp specialization")


if __name__ == "__main__":
    main()
