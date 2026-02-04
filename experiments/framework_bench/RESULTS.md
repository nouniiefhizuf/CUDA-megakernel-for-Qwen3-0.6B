# Framework Benchmark Results - Qwen3-0.6B

## Hardware
- GPU: RTX 3090 (24GB)
- CPU: AMD Threadripper
- CUDA: 12.1

## Benchmark Configuration
- Prompts: 3 test prompts
- Decode tokens per prompt: 100
- Temperature: 0.0 (greedy)

## Throughput Results

| Framework | TTFT (s) | Decode tok/s | Peak Memory | Speedup vs HF |
|-----------|----------|--------------|-------------|---------------|
| HuggingFace | 0.071 | 81 | 1.4 GB | 1.0x |
| **Megakernel** | **0.004** | **239** | 2.6 GB | **2.95x** |
| vLLM | 0.047 | 107 | ~2.0 GB | 1.32x |
| SGLang | - | - | - | Requires server |
| llama.cpp | - | - | - | Needs GGUF conversion |

## Power & Energy Efficiency

| Framework | Idle (W) | Avg (W) | Peak (W) | tok/s | tok/J | Efficiency vs HF |
|-----------|----------|---------|----------|-------|-------|------------------|
| HuggingFace | 169.1 | 185.3 | 190.2 | 58.4 | 0.31 | 1.0x |
| **Megakernel** | 172.1 | 200.2 | 228.3 | 156.6 | **0.78** | **2.48x** |
| vLLM | 170.8 | 195.4 | 208.8 | 107.3 | 0.55 | 1.74x |

## Key Findings

1. **Megakernel is 2.95x faster than HuggingFace** for decode throughput
2. **Megakernel is 2.48x more energy efficient** (0.78 tok/J vs 0.31 tok/J)
3. **TTFT is 17.75x faster** (4ms vs 71ms) due to fused kernel launch
4. vLLM achieves 1.74x energy efficiency over HuggingFace
5. Memory usage is higher for megakernel (2.6 GB vs 1.4 GB) due to:
   - Pre-allocated intermediate buffers
   - Full KV cache allocation upfront

## Quality Metrics

| Framework | KL Divergence | Perplexity | Argmax Match |
|-----------|---------------|------------|--------------|
| HuggingFace | 0.0 (ref) | 35.51 | 100% |
| Megakernel | N/A | N/A | 100% |

The megakernel produces identical argmax predictions to HuggingFace on test prompts.
Full KL divergence comparison requires exposing logits from the megakernel.

## Running Benchmarks

```bash
# Throughput benchmark
python experiments/framework_bench/benchmark_suite.py

# Quality metrics
python experiments/framework_bench/quality_metrics.py
```

## TODO

- [x] Run vLLM in isolated process (OOM with other models loaded)
- [ ] Convert model to GGUF for llama.cpp/Ollama
- [ ] Set up SGLang server for comparison
- [ ] Add TensorRT-LLM benchmark
- [ ] Add ExLlamaV2 benchmark
