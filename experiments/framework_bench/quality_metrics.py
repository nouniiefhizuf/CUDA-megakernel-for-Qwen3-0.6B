"""
Quality metrics for Qwen3-0.6B across frameworks.

Computes KL divergence and perplexity by comparing logits/probabilities.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_NAME = "Qwen/Qwen3-0.6B"
VOCAB_SIZE = 151936

TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, gradient descent is",
    "The capital of France is",
    "Write a function to compute fibonacci numbers:",
    "The meaning of life is",
]


@dataclasses.dataclass
class QualityResult:
    name: str
    supported: bool
    kl_div: Optional[float]
    perplexity: Optional[float]
    argmax_match: Optional[float]
    error: Optional[str] = None


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _format_float(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _format_markdown_table(results: List[QualityResult]) -> str:
    lines = [
        "| Framework | KL Divergence | Perplexity | Argmax Match | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        status = "ok" if r.supported else (r.error or "skipped")
        argmax_str = f"{r.argmax_match:.1%}" if r.argmax_match is not None else "n/a"
        lines.append(
            f"| {r.name} | {_format_float(r.kl_div)} | {_format_float(r.perplexity, 2)} | {argmax_str} | {status} |"
        )
    return "\n".join(lines)


def kl_divergence(ref_logits: torch.Tensor, tgt_logits: torch.Tensor) -> float:
    """Compute KL divergence: KL(P_ref || P_tgt) where P = softmax(logits)."""
    ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
    tgt_log_probs = F.log_softmax(tgt_logits.float(), dim=-1)
    ref_probs = ref_log_probs.exp()

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = (ref_probs * (ref_log_probs - tgt_log_probs)).sum(dim=-1).mean()
    return float(kl.item())


def perplexity_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Compute perplexity from logits and input_ids."""
    # logits shape: [seq_len, vocab_size] or [1, seq_len, vocab_size]
    if logits.dim() == 3:
        logits = logits.squeeze(0)

    # Shift for next-token prediction
    shift_logits = logits[:-1, :].float()
    shift_labels = input_ids[1:] if input_ids.dim() == 1 else input_ids[0, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(nll.mean()).item())


# =============================================================================
# HuggingFace Reference
# =============================================================================

def get_hf_logits(prompts: List[str]) -> Tuple[QualityResult, List[torch.Tensor], List[torch.Tensor]]:
    """Get HuggingFace reference logits."""
    print("\n" + "="*60)
    print("HUGGINGFACE REFERENCE")
    print("="*60)

    clear_gpu_memory()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        )
        model.eval()

        all_logits = []
        all_input_ids = []
        total_ppl = 0.0

        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                out = model(input_ids, use_cache=False)
            # Store logits for last position only (for next-token prediction comparison)
            all_logits.append(out.logits[:, -1, :].clone())
            all_input_ids.append(input_ids.clone())
            total_ppl += perplexity_from_logits(out.logits, input_ids)

        avg_ppl = total_ppl / len(prompts)

        del model
        clear_gpu_memory()

        print(f"  Perplexity: {avg_ppl:.2f}")
        return QualityResult("HuggingFace", True, 0.0, avg_ppl, 1.0), all_logits, all_input_ids

    except Exception as exc:
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return QualityResult("HuggingFace", False, None, None, None, str(exc)), [], []


# =============================================================================
# Megakernel
# =============================================================================

def get_megakernel_logits(prompts: List[str], ref_logits: List[torch.Tensor], ref_input_ids: List[torch.Tensor]) -> QualityResult:
    """Get megakernel logits and compare to HuggingFace."""
    print("\n" + "="*60)
    print("MEGAKERNEL")
    print("="*60)

    clear_gpu_memory()

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from chat import MegakernelChat, NUM_LAYERS, MAX_SEQ_LEN
        from transformers import AutoTokenizer

        print("Loading megakernel...")
        chat = MegakernelChat()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

        total_kl = 0.0
        total_ppl = 0.0
        matches = 0
        total = 0

        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...")
            chat.k_cache.zero_()
            chat.v_cache.zero_()

            input_ids = tokenizer.encode(prompt)

            # Process all tokens except the last one
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

            # Get logits for the last position
            token_id, logits = chat.kernel.decode_ldg_with_logits(
                input_ids[-1], chat.final_norm_weight, chat.lm_head_weight,
                chat.cos_table, chat.sin_table,
                chat.k_cache, chat.v_cache,
                chat.hidden_buffer, chat.g_activations, chat.g_residual,
                chat.g_q, chat.g_k, chat.g_v, chat.g_attn_out,
                chat.g_mlp_intermediate, chat.g_normalized,
                chat.block_max_vals, chat.block_max_idxs,
                NUM_LAYERS, len(input_ids) - 1, len(input_ids), MAX_SEQ_LEN,
            )

            # Compare logits
            mega_logits = logits.unsqueeze(0)  # [1, vocab_size]
            ref = ref_logits[i]  # [1, vocab_size]

            kl = kl_divergence(ref, mega_logits)
            total_kl += kl

            # Argmax comparison
            hf_pred = ref.argmax(dim=-1).item()
            mega_pred = mega_logits.argmax(dim=-1).item()
            if hf_pred == mega_pred:
                matches += 1
            total += 1

        avg_kl = total_kl / len(prompts)
        argmax_match = matches / total if total > 0 else 0.0

        del chat
        clear_gpu_memory()

        print(f"  KL Divergence: {avg_kl:.6f}")
        print(f"  Argmax Match: {argmax_match:.1%}")

        return QualityResult("Megakernel", True, avg_kl, None, argmax_match)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return QualityResult("Megakernel", False, None, None, None, str(exc))


# =============================================================================
# vLLM
# =============================================================================

def get_vllm_logits(prompts: List[str], ref_logits: List[torch.Tensor], ref_input_ids: List[torch.Tensor]) -> QualityResult:
    """Get vLLM logits and compare to HuggingFace."""
    print("\n" + "="*60)
    print("vLLM")
    print("="*60)

    clear_gpu_memory()

    try:
        from vllm import LLM, SamplingParams

        print("Loading vLLM...")
        llm = LLM(model=MODEL_NAME, dtype="bfloat16", gpu_memory_utilization=0.8)

        # vLLM doesn't easily expose logits, so we compare argmax only
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1)

        matches = 0
        total = 0

        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...")
            outputs = llm.generate([prompt], params, use_tqdm=False)

            if outputs and outputs[0].outputs:
                vllm_pred = outputs[0].outputs[0].token_ids[0] if outputs[0].outputs[0].token_ids else -1
                hf_pred = ref_logits[i].argmax(dim=-1).item()

                if vllm_pred == hf_pred:
                    matches += 1
            total += 1

        argmax_match = matches / total if total > 0 else 0.0

        del llm
        clear_gpu_memory()

        print(f"  Argmax Match: {argmax_match:.1%}")

        # KL divergence not available without full logits exposure
        return QualityResult("vLLM", True, None, None, argmax_match, "logits not exposed")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return QualityResult("vLLM", False, None, None, None, str(exc))


# =============================================================================
# SGLang
# =============================================================================

def get_sglang_logits(prompts: List[str], ref_logits: List[torch.Tensor], ref_input_ids: List[torch.Tensor]) -> QualityResult:
    """Get SGLang predictions and compare to HuggingFace."""
    print("\n" + "="*60)
    print("SGLANG")
    print("="*60)

    clear_gpu_memory()

    try:
        import sglang as sgl
        from sglang import RuntimeEndpoint

        print("Starting SGLang server...")
        # SGLang requires running a server, which is complex for a benchmark
        # For now, skip SGLang in automated benchmarks
        return QualityResult("SGLang", False, None, None, None, "requires server setup")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return QualityResult("SGLang", False, None, None, None, str(exc))


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print("="*60)
    print("QUALITY METRICS - Qwen3-0.6B")
    print("="*60)
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Vocab size: {VOCAB_SIZE}")

    # Get HuggingFace reference
    hf_result, ref_logits, ref_input_ids = get_hf_logits(TEST_PROMPTS)
    results = [hf_result]

    if not hf_result.supported:
        print("ERROR: HuggingFace reference failed, cannot continue")
        return 1

    # Compare each framework sequentially
    results.append(get_megakernel_logits(TEST_PROMPTS, ref_logits, ref_input_ids))
    results.append(get_vllm_logits(TEST_PROMPTS, ref_logits, ref_input_ids))
    results.append(get_sglang_logits(TEST_PROMPTS, ref_logits, ref_input_ids))

    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print()
    print(_format_markdown_table(results))

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "quality_results.md")

    with open(output_path, "w") as f:
        f.write("# Quality Metrics - Qwen3-0.6B\n\n")
        f.write(f"**Test prompts**: {len(TEST_PROMPTS)}\n")
        f.write(f"**Vocab size**: {VOCAB_SIZE}\n\n")
        f.write("## Results\n\n")
        f.write(_format_markdown_table(results))
        f.write("\n\n## Metrics Explanation\n\n")
        f.write("- **KL Divergence**: KL(P_HF || P_framework) measures how different the probability distributions are. Lower is better. 0.0 means identical.\n")
        f.write("- **Perplexity**: Measures model uncertainty. Lower is better.\n")
        f.write("- **Argmax Match**: Percentage of prompts where the predicted next token matches HuggingFace. 100% means identical greedy outputs.\n")

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
