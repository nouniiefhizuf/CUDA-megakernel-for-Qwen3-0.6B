"""Framework benchmark suite for Qwen3-0.6B."""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from typing import List, Optional

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen3-0.6B"
DECODE_TOKENS = 100
DEFAULT_PROMPTS = [
    "Explain the concept of overfitting in machine learning.",
    "Write a short story about a lost robot.",
    "List three benefits of GPU acceleration for deep learning.",
]


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    supported: bool
    ttft_s: Optional[float]
    decode_toks_per_s: Optional[float]
    memory_bytes: Optional[int]
    error: Optional[str] = None


def _torch_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _max_memory_allocated() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024 ** 2):.1f} MB"


def _format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _format_markdown_table(results: List[BenchmarkResult]) -> str:
    lines = [
        "| Framework | TTFT (s) | Decode tok/s | Peak mem | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        status = "ok" if r.supported else (r.error or "skipped")
        lines.append(
            f"| {r.name} | {_format_float(r.ttft_s)} | {_format_float(r.decode_toks_per_s, 2)} | "
            f"{_format_bytes(r.memory_bytes)} | {status} |"
        )
    return "\n".join(lines)


def _hf_load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def _hf_benchmark(prompts: List[str]) -> BenchmarkResult:
    if not torch.cuda.is_available():
        return BenchmarkResult("HuggingFace", False, None, None, None, "cuda not available")

    try:
        _reset_peak_memory()
        model, tokenizer = _hf_load()
    except Exception as exc:
        return BenchmarkResult("HuggingFace", False, None, None, None, f"load error: {exc}")

    ttft_values: List[float] = []
    total_tokens = 0
    total_time = 0.0

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

            _torch_sync()
            t0 = time.perf_counter()
            out = model(input_ids, use_cache=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            _torch_sync()
            t1 = time.perf_counter()
            ttft_values.append(t1 - t0)

            past = out.past_key_values
            generated = 1

            _torch_sync()
            t_decode_start = time.perf_counter()
            for _ in range(DECODE_TOKENS - 1):
                out = model(next_token.unsqueeze(0), use_cache=True, past_key_values=past)
                next_token = out.logits[:, -1, :].argmax(dim=-1)
                past = out.past_key_values
                generated += 1
            _torch_sync()
            t_decode_end = time.perf_counter()

            total_tokens += generated
            total_time += (t_decode_end - t_decode_start) + (t1 - t0)

    memory_bytes = _max_memory_allocated()

    del model
    torch.cuda.empty_cache()

    ttft_avg = sum(ttft_values) / len(ttft_values) if ttft_values else None
    decode_toks_per_s = (total_tokens / total_time) if total_time > 0 else None

    return BenchmarkResult("HuggingFace", True, ttft_avg, decode_toks_per_s, memory_bytes)


def _megakernel_benchmark(prompts: List[str]) -> BenchmarkResult:
    if not torch.cuda.is_available():
        return BenchmarkResult("Megakernel", False, None, None, None, "cuda not available")

    try:
        _reset_peak_memory()
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from chat import MegakernelChat, NUM_LAYERS, MAX_SEQ_LEN
    except Exception as exc:
        return BenchmarkResult("Megakernel", False, None, None, None, f"import error: {exc}")

    try:
        chat = MegakernelChat()
    except Exception as exc:
        return BenchmarkResult("Megakernel", False, None, None, None, f"init error: {exc}")

    ttft_values: List[float] = []
    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        chat.k_cache.zero_()
        chat.v_cache.zero_()
        input_ids = chat.tokenizer.encode(prompt)

        # Prefill
        for pos, token_id in enumerate(input_ids[:-1]):
            chat.kernel.decode_ldg(
                token_id, chat.final_norm_weight, chat.lm_head_weight,
                chat.cos_table, chat.sin_table,
                chat.k_cache, chat.v_cache,
                chat.hidden_buffer, chat.g_activations, chat.g_residual,
                chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
                chat.g_mlp_intermediate, chat.g_normalized,
                chat.block_max_vals, chat.block_max_idxs,
                NUM_LAYERS, pos, pos + 1, MAX_SEQ_LEN,
            )

        current_pos = len(input_ids) - 1
        current_token = input_ids[-1]

        # First token (TTFT)
        _torch_sync()
        t0 = time.perf_counter()
        next_token = chat.kernel.decode_ldg(
            current_token, chat.final_norm_weight, chat.lm_head_weight,
            chat.cos_table, chat.sin_table,
            chat.k_cache, chat.v_cache,
            chat.hidden_buffer, chat.g_activations, chat.g_residual,
            chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
            chat.g_mlp_intermediate, chat.g_normalized,
            chat.block_max_vals, chat.block_max_idxs,
            NUM_LAYERS, current_pos, current_pos + 1, MAX_SEQ_LEN,
        )
        _torch_sync()
        t1 = time.perf_counter()
        ttft_values.append(t1 - t0)

        generated = 1
        current_pos += 1

        # Decode remaining tokens
        _torch_sync()
        t_decode_start = time.perf_counter()
        for _ in range(DECODE_TOKENS - 1):
            next_token = chat.kernel.decode_ldg(
                next_token, chat.final_norm_weight, chat.lm_head_weight,
                chat.cos_table, chat.sin_table,
                chat.k_cache, chat.v_cache,
                chat.hidden_buffer, chat.g_activations, chat.g_residual,
                chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
                chat.g_mlp_intermediate, chat.g_normalized,
                chat.block_max_vals, chat.block_max_idxs,
                NUM_LAYERS, current_pos, current_pos + 1, MAX_SEQ_LEN,
            )
            current_pos += 1
            generated += 1
        _torch_sync()
        t_decode_end = time.perf_counter()

        total_tokens += generated
        total_time += (t_decode_end - t_decode_start) + (t1 - t0)

    memory_bytes = _max_memory_allocated()

    del chat
    torch.cuda.empty_cache()

    ttft_avg = sum(ttft_values) / len(ttft_values) if ttft_values else None
    decode_toks_per_s = (total_tokens / total_time) if total_time > 0 else None

    return BenchmarkResult("Megakernel", True, ttft_avg, decode_toks_per_s, memory_bytes)


def _vllm_benchmark(prompts: List[str]) -> BenchmarkResult:
    if not torch.cuda.is_available():
        return BenchmarkResult("vLLM", False, None, None, None, "cuda not available")

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        return BenchmarkResult("vLLM", False, None, None, None, f"import error: {exc}")

    try:
        _reset_peak_memory()
        llm = LLM(model=MODEL_NAME, dtype="bfloat16", download_dir=None)
    except Exception as exc:
        return BenchmarkResult("vLLM", False, None, None, None, f"init error: {exc}")

    ttft_values: List[float] = []
    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        params_ttft = SamplingParams(temperature=0.0, max_tokens=1)
        _torch_sync()
        t0 = time.perf_counter()
        llm.generate([prompt], params_ttft, use_tqdm=False)
        _torch_sync()
        t1 = time.perf_counter()
        ttft_values.append(t1 - t0)

        params_full = SamplingParams(temperature=0.0, max_tokens=DECODE_TOKENS)
        _torch_sync()
        t2 = time.perf_counter()
        llm.generate([prompt], params_full, use_tqdm=False)
        _torch_sync()
        t3 = time.perf_counter()

        total_tokens += DECODE_TOKENS
        total_time += (t3 - t2)

    memory_bytes = _max_memory_allocated()

    del llm
    torch.cuda.empty_cache()

    ttft_avg = sum(ttft_values) / len(ttft_values) if ttft_values else None
    decode_toks_per_s = (total_tokens / total_time) if total_time > 0 else None

    return BenchmarkResult("vLLM", True, ttft_avg, decode_toks_per_s, memory_bytes)


def _sglang_benchmark(prompts: List[str]) -> BenchmarkResult:
    if not torch.cuda.is_available():
        return BenchmarkResult("SGLang", False, None, None, None, "cuda not available")

    try:
        import sglang as sgl
        from sglang import RuntimeEndpoint
    except Exception as exc:
        return BenchmarkResult("SGLang", False, None, None, None, f"import error: {exc}")

    # SGLang requires a running server, skip for now
    return BenchmarkResult("SGLang", False, None, None, None, "requires server setup")


def _llamacpp_benchmark(prompts: List[str]) -> BenchmarkResult:
    """Benchmark llama.cpp via llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except Exception as exc:
        return BenchmarkResult("llama.cpp", False, None, None, None, f"import error: {exc}")

    # Check for GGUF model
    gguf_path = os.path.expanduser("~/.cache/gguf/qwen3-0.6b.gguf")
    if not os.path.exists(gguf_path):
        return BenchmarkResult("llama.cpp", False, None, None, None, f"GGUF not found at {gguf_path}")

    try:
        _reset_peak_memory()
        llm = Llama(model_path=gguf_path, n_ctx=512, n_gpu_layers=-1, verbose=False)
    except Exception as exc:
        return BenchmarkResult("llama.cpp", False, None, None, None, f"init error: {exc}")

    ttft_values: List[float] = []
    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        _torch_sync()
        t0 = time.perf_counter()
        output = llm(prompt, max_tokens=DECODE_TOKENS, temperature=0.0)
        _torch_sync()
        t1 = time.perf_counter()

        generated = output["usage"]["completion_tokens"]
        total_tokens += generated
        total_time += (t1 - t0)
        # TTFT not easily measurable with this API
        ttft_values.append(t1 - t0)  # Approximation

    memory_bytes = _max_memory_allocated()

    del llm
    torch.cuda.empty_cache()

    ttft_avg = sum(ttft_values) / len(ttft_values) if ttft_values else None
    decode_toks_per_s = (total_tokens / total_time) if total_time > 0 else None

    return BenchmarkResult("llama.cpp", True, ttft_avg, decode_toks_per_s, memory_bytes)


def main() -> int:
    prompts = DEFAULT_PROMPTS

    print("=" * 60)
    print("FRAMEWORK BENCHMARK SUITE - Qwen3-0.6B")
    print("=" * 60)
    print(f"Prompts: {len(prompts)}")
    print(f"Decode tokens per prompt: {DECODE_TOKENS}")
    print()

    results = []

    print("Benchmarking HuggingFace...")
    results.append(_hf_benchmark(prompts))

    print("Benchmarking Megakernel...")
    results.append(_megakernel_benchmark(prompts))

    print("Benchmarking vLLM...")
    results.append(_vllm_benchmark(prompts))

    print("Benchmarking SGLang...")
    results.append(_sglang_benchmark(prompts))

    print("Benchmarking llama.cpp...")
    results.append(_llamacpp_benchmark(prompts))

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(_format_markdown_table(results))

    print()
    tested = [r.name for r in results if r.supported]
    skipped = [r.name for r in results if not r.supported]

    print("Tested:", ", ".join(tested) if tested else "none")
    print("Skipped:", ", ".join(skipped) if skipped else "none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
