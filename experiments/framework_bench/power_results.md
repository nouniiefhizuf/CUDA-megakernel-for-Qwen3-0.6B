# GPU Power Benchmark Results

**GPU**: NVIDIA GeForce RTX 3090
**Power Limit**: 420 W
**Memory**: 24576 MB
**Prompt**: "Write a detailed essay about lobsters, covering th..."
**Max Tokens**: 200

## Results

| Framework | Idle (W) | Avg (W) | Peak (W) | Time (s) | Tokens | tok/s | tok/J | Status |
|-----------|----------|---------|----------|----------|--------|-------|-------|--------|
| HuggingFace | 169.1 | 185.3 | 190.2 | 3.43 | 200 | 58.4 | 0.31 | ok |
| Megakernel | 172.1 | 200.2 | 228.3 | 1.15 | 180 | 156.6 | 0.78 | ok |
| vLLM | 170.8 | 195.4 | 208.8 | 1.86 | 200 | 107.3 | 0.55 | ok |

## Analysis

**Baseline**: HuggingFace

- **Megakernel**: 2.48x more energy efficient than HuggingFace
- **vLLM**: 1.74x more energy efficient than HuggingFace