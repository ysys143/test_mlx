"""Main benchmark runner.

Usage:
    uv run python benchmark.py [--backends mlx ollama llamacpp vllm gptoss]

Results saved to results/benchmark_<timestamp>.json and results/benchmark_report.md
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent))
from config import PROMPT, MAX_TOKENS, RUNS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_BACKENDS = ["mlx", "ollama", "llamacpp", "vllm", "gptoss"]


def _run_backend(name: str) -> list[dict]:
    print(f"\n[{name.upper()}] starting...")
    try:
        if name == "mlx":
            from backends.bench_mlx import run
            return [run(PROMPT, MAX_TOKENS) for _ in range(RUNS)]
        elif name == "ollama":
            from backends.bench_ollama import run
            return [run(PROMPT, MAX_TOKENS) for _ in range(RUNS)]
        elif name == "llamacpp":
            from backends.bench_llamacpp import run
            return [run(PROMPT, MAX_TOKENS) for _ in range(RUNS)]
        elif name == "vllm":
            # vllm returns [mlx_result, paged_result] per call - run once (server startup is expensive)
            from backends.bench_vllm_metal import run
            return run(PROMPT, MAX_TOKENS)  # already runs both modes
        elif name == "gptoss":
            from backends.bench_gptoss import run
            return [run(PROMPT, MAX_TOKENS) for _ in range(RUNS)]
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return []


def _average(results: list[dict]) -> dict:
    if not results:
        return {}
    if len(results) == 1:
        return results[0]
    base = dict(results[0])
    for key in ("tokens_per_sec", "ttft_ms", "total_sec"):
        vals = [r[key] for r in results if key in r]
        base[key] = mean(vals)
        base[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0
    return base


def _format_table(rows: list[dict]) -> str:
    headers = ["Backend", "tokens/sec", "TTFT (ms)", "total_sec"]
    col_w = [max(len(h), max(len(str(r.get(k, "N/A"))) for r in rows))
             for h, k in zip(headers, ["backend", "tokens_per_sec", "ttft_ms", "total_sec"])]
    col_w = [max(w, len(h)) + 2 for w, h in zip(col_w, headers)]

    def row_str(vals):
        return "| " + " | ".join(str(v).ljust(w - 2) for v, w in zip(vals, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * (w - 2) for w in col_w) + "-|"
    lines = [
        row_str(headers),
        sep,
    ]
    for r in rows:
        tps = f"{r.get('tokens_per_sec', 0):.2f}" if r else "ERR"
        ttft = f"{r.get('ttft_ms', 0):.0f}" if r else "ERR"
        tsec = f"{r.get('total_sec', 0):.2f}" if r else "ERR"
        lines.append(row_str([r.get("backend", "?"), tps, ttft, tsec]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", choices=ALL_BACKENDS, default=ALL_BACKENDS)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_raw: dict[str, list] = {}
    averaged: list[dict] = []

    for backend in args.backends:
        raw = _run_backend(backend)
        all_raw[backend] = [r for r in raw if r]

        # group by backend name (vllm returns 2 different backends)
        groups: dict[str, list] = {}
        for r in all_raw[backend]:
            groups.setdefault(r["backend"], []).append(r)
        for group_results in groups.values():
            avg = _average(group_results)
            averaged.append(avg)
            print(f"  -> {avg.get('backend')}: {avg.get('tokens_per_sec', 0):.2f} tok/s  TTFT={avg.get('ttft_ms', 0):.0f}ms")

    # save raw JSON
    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"timestamp": timestamp, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "runs": RUNS, "results": all_raw}, f, indent=2)

    # save markdown report
    report_path = RESULTS_DIR / "benchmark_report.md"
    table = _format_table(averaged)
    report = f"""# Benchmark Report: Qwen3.5-9B + GPT-OSS

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Prompt**: {PROMPT[:80]}...
**Max tokens**: {MAX_TOKENS}
**Runs averaged**: {RUNS}

## Results

{table}

## Notes

- MLX: 4-bit quantized, Unified Memory, Apple Silicon optimized
- vLLM Metal (MLX path): MLX-managed KV cache, no paged attention
- vLLM Metal (Paged): VLLM_METAL_USE_PAGED_ATTENTION=1 (experimental)
- Ollama: llama.cpp-based server with HTTP overhead
- llama.cpp Metal: direct Metal GPU, Q4_K_M GGUF, ngl=99
- GPT-OSS-20B: MoE model (~3.6B active params/token), openharmony-mlx
"""
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n[DONE] JSON: {json_path}")
    print(f"       Report: {report_path}")
    print(f"\n{table}")


if __name__ == "__main__":
    main()
