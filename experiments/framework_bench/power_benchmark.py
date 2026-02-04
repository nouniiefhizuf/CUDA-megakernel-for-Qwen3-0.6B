"""
Power consumption benchmark for Qwen3-0.6B inference.

Measures peak and average GPU power during generation.
Each framework runs in isolation to avoid OOM.
"""

import os
import sys
import time
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

# Ensure offline mode for HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"
TEST_PROMPT = "Write a detailed essay about lobsters, covering their biology, habitat, and importance to marine ecosystems."
MAX_TOKENS = 200


@dataclass
class PowerReading:
    timestamp: float
    power_watts: float


@dataclass
class PowerResult:
    framework: str
    supported: bool
    peak_power_w: Optional[float]
    avg_power_w: Optional[float]
    idle_power_w: Optional[float]
    generation_time_s: Optional[float]
    tokens_generated: Optional[int]
    tok_per_joule: Optional[float]
    error: Optional[str] = None


class PowerMonitor:
    """Monitor GPU power using nvidia-smi."""

    def __init__(self, gpu_id: int = 0, sample_interval_ms: int = 50):
        self.gpu_id = gpu_id
        self.sample_interval_ms = sample_interval_ms
        self.readings: List[PowerReading] = []
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def _read_power(self) -> float:
        """Read current GPU power in watts."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True, text=True, timeout=1
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _monitor_loop(self):
        """Background thread to sample power readings."""
        while not self._stop:
            power = self._read_power()
            self.readings.append(PowerReading(time.perf_counter(), power))
            time.sleep(self.sample_interval_ms / 1000.0)

    def start(self):
        """Start monitoring."""
        self.readings = []
        self._stop = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, float]:
        """Stop monitoring and return (peak_power, avg_power)."""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self.readings:
            return 0.0, 0.0

        powers = [r.power_watts for r in self.readings]
        return max(powers), sum(powers) / len(powers)

    def get_idle_power(self, duration_s: float = 2.0) -> float:
        """Measure idle power for a duration."""
        readings = []
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            readings.append(self._read_power())
            time.sleep(0.1)
        return sum(readings) / len(readings) if readings else 0.0


def get_gpu_info() -> dict:
    """Get GPU info from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,power.limit,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(", ")
        return {
            "name": parts[0] if len(parts) > 0 else "Unknown",
            "power_limit_w": float(parts[1].replace(" W", "")) if len(parts) > 1 else 0,
            "memory_mb": float(parts[2].replace(" MiB", "")) if len(parts) > 2 else 0,
        }
    except Exception as e:
        return {"name": "Unknown", "power_limit_w": 0, "memory_mb": 0, "error": str(e)}


def clear_gpu_memory():
    """Clear GPU memory between benchmarks."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(2)  # Let GPU settle


def benchmark_huggingface(monitor: PowerMonitor) -> PowerResult:
    """Benchmark HuggingFace Transformers."""
    print("\n" + "="*60)
    print("BENCHMARKING: HuggingFace Transformers")
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

        # Warmup
        print("Warming up...")
        input_ids = tokenizer(TEST_PROMPT, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

        # Measure idle power
        print("Measuring idle power...")
        idle_power = monitor.get_idle_power(2.0)

        # Benchmark with power monitoring
        print(f"Generating {MAX_TOKENS} tokens...")
        input_ids = tokenizer(TEST_PROMPT, return_tensors="pt").input_ids.to("cuda")

        monitor.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=MAX_TOKENS, do_sample=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        tokens = output.shape[1] - input_ids.shape[1]
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        del model, tokenizer
        clear_gpu_memory()

        return PowerResult(
            framework="HuggingFace",
            supported=True,
            peak_power_w=peak_power,
            avg_power_w=avg_power,
            idle_power_w=idle_power,
            generation_time_s=gen_time,
            tokens_generated=tokens,
            tok_per_joule=tok_per_joule,
        )

    except Exception as e:
        clear_gpu_memory()
        return PowerResult("HuggingFace", False, None, None, None, None, None, None, str(e))


def benchmark_megakernel(monitor: PowerMonitor) -> PowerResult:
    """Benchmark our custom megakernel."""
    print("\n" + "="*60)
    print("BENCHMARKING: Megakernel")
    print("="*60)

    clear_gpu_memory()

    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, project_root)
        from chat import MegakernelChat, NUM_LAYERS, MAX_SEQ_LEN

        print("Loading megakernel...")
        chat = MegakernelChat()

        # Warmup
        print("Warming up...")
        _ = chat.generate(TEST_PROMPT, max_new_tokens=10, show_speed=False)
        torch.cuda.synchronize()

        # Measure idle power
        print("Measuring idle power...")
        idle_power = monitor.get_idle_power(2.0)

        # Benchmark with power monitoring
        print(f"Generating {MAX_TOKENS} tokens...")

        monitor.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        output = chat.generate(TEST_PROMPT, max_new_tokens=MAX_TOKENS, show_speed=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        # Count tokens from output
        tokens = len(chat.tokenizer.encode(output)) - len(chat.tokenizer.encode(TEST_PROMPT))
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        del chat
        clear_gpu_memory()

        return PowerResult(
            framework="Megakernel",
            supported=True,
            peak_power_w=peak_power,
            avg_power_w=avg_power,
            idle_power_w=idle_power,
            generation_time_s=gen_time,
            tokens_generated=tokens,
            tok_per_joule=tok_per_joule,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return PowerResult("Megakernel", False, None, None, None, None, None, None, str(e))


def benchmark_vllm(monitor: PowerMonitor) -> PowerResult:
    """Benchmark vLLM."""
    print("\n" + "="*60)
    print("BENCHMARKING: vLLM")
    print("="*60)

    clear_gpu_memory()

    try:
        from vllm import LLM, SamplingParams

        print("Loading vLLM...")
        llm = LLM(model=MODEL_NAME, dtype="bfloat16", gpu_memory_utilization=0.8)

        # Warmup
        print("Warming up...")
        params = SamplingParams(temperature=0.0, max_tokens=10)
        _ = llm.generate([TEST_PROMPT], params, use_tqdm=False)

        # Measure idle power
        print("Measuring idle power...")
        idle_power = monitor.get_idle_power(2.0)

        # Benchmark with power monitoring
        print(f"Generating {MAX_TOKENS} tokens...")
        params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

        monitor.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = llm.generate([TEST_PROMPT], params, use_tqdm=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_power, avg_power = monitor.stop()

        gen_time = end - start
        tokens = len(outputs[0].outputs[0].token_ids) if outputs else MAX_TOKENS
        energy_joules = avg_power * gen_time
        tok_per_joule = tokens / energy_joules if energy_joules > 0 else 0

        del llm
        clear_gpu_memory()

        return PowerResult(
            framework="vLLM",
            supported=True,
            peak_power_w=peak_power,
            avg_power_w=avg_power,
            idle_power_w=idle_power,
            generation_time_s=gen_time,
            tokens_generated=tokens,
            tok_per_joule=tok_per_joule,
        )

    except Exception as e:
        clear_gpu_memory()
        return PowerResult("vLLM", False, None, None, None, None, None, None, str(e))


def format_results(results: List[PowerResult], gpu_info: dict) -> str:
    """Format results as markdown table."""
    lines = [
        "# GPU Power Benchmark Results",
        "",
        f"**GPU**: {gpu_info['name']}",
        f"**Power Limit**: {gpu_info['power_limit_w']:.0f} W",
        f"**Memory**: {gpu_info['memory_mb']:.0f} MB",
        f"**Prompt**: \"{TEST_PROMPT[:50]}...\"",
        f"**Max Tokens**: {MAX_TOKENS}",
        "",
        "## Results",
        "",
        "| Framework | Idle (W) | Avg (W) | Peak (W) | Time (s) | Tokens | tok/s | tok/J | Status |",
        "|-----------|----------|---------|----------|----------|--------|-------|-------|--------|",
    ]

    for r in results:
        if r.supported:
            tok_s = r.tokens_generated / r.generation_time_s if r.generation_time_s else 0
            lines.append(
                f"| {r.framework} | {r.idle_power_w:.1f} | {r.avg_power_w:.1f} | {r.peak_power_w:.1f} | "
                f"{r.generation_time_s:.2f} | {r.tokens_generated} | {tok_s:.1f} | {r.tok_per_joule:.2f} | ok |"
            )
        else:
            lines.append(f"| {r.framework} | - | - | - | - | - | - | - | {r.error or 'skipped'} |")

    lines.extend([
        "",
        "## Analysis",
        "",
    ])

    # Find supported results for comparison
    supported = [r for r in results if r.supported]
    if len(supported) >= 2:
        # Compare energy efficiency
        baseline = next((r for r in supported if r.framework == "HuggingFace"), supported[0])
        lines.append(f"**Baseline**: {baseline.framework}")
        lines.append("")

        for r in supported:
            if r != baseline and r.tok_per_joule and baseline.tok_per_joule:
                efficiency_ratio = r.tok_per_joule / baseline.tok_per_joule
                lines.append(f"- **{r.framework}**: {efficiency_ratio:.2f}x more energy efficient than {baseline.framework}")

    return "\n".join(lines)


def main():
    print("="*60)
    print("GPU POWER BENCHMARK - Qwen3-0.6B")
    print("="*60)

    # Get GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Power Limit: {gpu_info['power_limit_w']:.0f} W")
    print(f"Memory: {gpu_info['memory_mb']:.0f} MB")

    # Create power monitor
    monitor = PowerMonitor(gpu_id=0, sample_interval_ms=50)

    # Run benchmarks sequentially
    results = []

    results.append(benchmark_huggingface(monitor))
    results.append(benchmark_megakernel(monitor))
    results.append(benchmark_vllm(monitor))

    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    output = format_results(results, gpu_info)
    print(output)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "power_results.md")
    with open(output_path, "w") as f:
        f.write(output)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
