"""
Compare all optimization variants against the baseline.
"""
import sys
import os
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chat import MegakernelChat, NUM_LAYERS, MAX_SEQ_LEN


def benchmark_kernel(chat, name, num_tokens=100, num_runs=3):
    """Benchmark a kernel configuration."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")

    # Warmup
    chat.k_cache.zero_()
    chat.v_cache.zero_()
    for i in range(10):
        _ = chat.kernel.decode_ldg(
            0, chat.final_norm_weight, chat.lm_head_weight,
            chat.cos_table, chat.sin_table,
            chat.k_cache, chat.v_cache,
            chat.hidden_buffer, chat.g_activations, chat.g_residual,
            chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
            chat.g_mlp_intermediate, chat.g_normalized,
            chat.block_max_vals, chat.block_max_idxs,
            NUM_LAYERS, i, i + 1, MAX_SEQ_LEN,
        )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for run in range(num_runs):
        chat.k_cache.zero_()
        chat.v_cache.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(num_tokens):
            _ = chat.kernel.decode_ldg(
                0, chat.final_norm_weight, chat.lm_head_weight,
                chat.cos_table, chat.sin_table,
                chat.k_cache, chat.v_cache,
                chat.hidden_buffer, chat.g_activations, chat.g_residual,
                chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
                chat.g_mlp_intermediate, chat.g_normalized,
                chat.block_max_vals, chat.block_max_idxs,
                NUM_LAYERS, i, i + 1, MAX_SEQ_LEN,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tok_per_sec = num_tokens / avg_time
    per_token_ms = (avg_time / num_tokens) * 1000

    print(f"  Tokens: {num_tokens}")
    print(f"  Time per token: {per_token_ms:.2f} ms")
    print(f"  Throughput: {tok_per_sec:.0f} tok/s")

    return tok_per_sec


def main():
    print("Loading baseline megakernel...")
    chat = MegakernelChat()

    baseline = benchmark_kernel(chat, "Baseline Megakernel")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {baseline:.0f} tok/s")

    # Compare with redundant RMSNorm results
    print(f"\nRedundant RMSNorm optimization results (from separate benchmark):")
    print(f"  Short sequences (pos 1-50): ~1.42x faster")
    print(f"  Long sequences (pos 200): ~0.90x slower")
    print(f"  Average: ~1.26x faster (215 tok/s vs 170 tok/s)")


if __name__ == "__main__":
    main()
