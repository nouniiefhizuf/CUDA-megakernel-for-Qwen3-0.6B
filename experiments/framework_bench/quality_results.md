# Quality Metrics - Qwen3-0.6B

**Test prompts**: 5
**Vocab size**: 151936

## Results

| Framework | KL Divergence | Perplexity | Argmax Match | Status |
|---|---:|---:|---:|---|
| HuggingFace | 0.000000 | 96.55 | 100.0% | ok |
| Megakernel | 0.000725 | n/a | 100.0% | ok |
| vLLM | n/a | n/a | 100.0% | ok |

## Metrics Explanation

- **KL Divergence**: KL(P_HF || P_framework) measures how different the probability distributions are. Lower is better. 0.0 means identical.
- **Perplexity**: Measures model uncertainty. Lower is better.
- **Argmax Match**: Percentage of prompts where the predicted next token matches HuggingFace. 100% means identical greedy outputs.
