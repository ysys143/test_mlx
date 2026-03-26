"""Concurrency throughput benchmark.

Sends N simultaneous requests to server backends (Ollama, vLLM Metal)
and measures aggregate tokens/sec at each concurrency level.

Usage:
    uv run python backends/bench_concurrency.py
    uv run python backends/bench_concurrency.py --backend ollama --levels 1,2,4,8
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROMPT, MAX_TOKENS

CONCURRENCY_LEVELS = [1, 2, 4, 8]

# Backend endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
VLLM_URL = "http://localhost:8765/v1/completions"
VLLM_MODEL = "Qwen/Qwen3.5-9B"


@dataclass
class RequestResult:
    tokens: int
    latency_sec: float
    ttft_ms: float
    success: bool
    error: str = ""


def _ollama_request(prompt: str, max_tokens: int) -> RequestResult:
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": "qwen3.5:9b", "prompt": prompt, "stream": True,
                  "options": {"num_predict": max_tokens}},
            stream=True, timeout=300,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if data.get("response"):
                token_count += 1
            if data.get("done"):
                # Ollama reports eval_count directly
                token_count = data.get("eval_count", token_count)
                break
        end = time.perf_counter()
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        return RequestResult(tokens=token_count, latency_sec=end - start,
                             ttft_ms=ttft, success=True)
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(tokens=0, latency_sec=end - start,
                             ttft_ms=0, success=False, error=str(e))


def _vllm_request(prompt: str, max_tokens: int) -> RequestResult:
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    try:
        with requests.post(
            VLLM_URL,
            json={"model": VLLM_MODEL, "prompt": prompt,
                  "max_tokens": max_tokens, "stream": True},
            stream=True, timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunk = json.loads(line[6:])
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        token_count += 1
        end = time.perf_counter()
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        return RequestResult(tokens=token_count, latency_sec=end - start,
                             ttft_ms=ttft, success=True)
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(tokens=0, latency_sec=end - start,
                             ttft_ms=0, success=False, error=str(e))


def _check_server(url: str, is_ollama: bool = False) -> bool:
    try:
        check = "http://localhost:11434" if is_ollama else "http://localhost:8765/health"
        requests.get(check, timeout=2)
        return True
    except Exception:
        return False


def run_concurrency_level(backend: str, n: int, prompt: str, max_tokens: int) -> dict:
    """Run N simultaneous requests and collect aggregate stats."""
    fn = _ollama_request if backend == "ollama" else _vllm_request

    wall_start = time.perf_counter()
    results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(fn, prompt, max_tokens) for _ in range(n)]
        for f in as_completed(futures):
            results.append(f.result())

    wall_end = time.perf_counter()
    wall_sec = wall_end - wall_start

    successful = [r for r in results if r.success]
    total_tokens = sum(r.tokens for r in successful)
    avg_latency = sum(r.latency_sec for r in successful) / len(successful) if successful else 0
    avg_ttft = sum(r.ttft_ms for r in successful) / len(successful) if successful else 0

    return {
        "backend": backend,
        "concurrency": n,
        "requests": n,
        "successful": len(successful),
        "total_tokens": total_tokens,
        "wall_sec": round(wall_sec, 2),
        "aggregate_tok_per_sec": round(total_tokens / wall_sec, 2) if wall_sec > 0 else 0,
        "avg_latency_sec": round(avg_latency, 2),
        "avg_ttft_ms": round(avg_ttft, 1),
    }


def run(backend: str = "both", levels: list[int] = None, prompt: str = PROMPT,
        max_tokens: int = MAX_TOKENS) -> list[dict]:
    if levels is None:
        levels = CONCURRENCY_LEVELS

    backends = []
    if backend in ("both", "ollama"):
        if _check_server("", is_ollama=True):
            backends.append("ollama")
        else:
            print("[WARN] Ollama not running — skipping", file=sys.stderr)
    if backend in ("both", "vllm"):
        if _check_server("http://localhost:8765/health"):
            backends.append("vllm")
        else:
            print("[WARN] vLLM Metal not running (start with: source ~/.venv-vllm-metal/bin/activate && vllm serve Qwen/Qwen3.5-9B --port 8765)", file=sys.stderr)

    results = []
    for b in backends:
        print(f"\n=== {b.upper()} ===", file=sys.stderr)
        # warm-up
        print(f"  Warm-up (concurrency=1)...", file=sys.stderr)
        fn = _ollama_request if b == "ollama" else _vllm_request
        fn(prompt, 10)

        for n in levels:
            print(f"  concurrency={n}...", file=sys.stderr)
            r = run_concurrency_level(b, n, prompt, max_tokens)
            results.append(r)
            print(f"    {r['aggregate_tok_per_sec']} tok/s aggregate, "
                  f"avg latency {r['avg_latency_sec']}s", file=sys.stderr)

    return results


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Backend':<10} {'Concurrency':>11} {'Agg tok/s':>10} "
          f"{'Avg lat(s)':>11} {'Avg TTFT(ms)':>13} {'Total tok':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['backend']:<10} {r['concurrency']:>11} "
              f"{r['aggregate_tok_per_sec']:>10.1f} "
              f"{r['avg_latency_sec']:>11.2f} "
              f"{r['avg_ttft_ms']:>13.0f} "
              f"{r['total_tokens']:>10}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="both",
                        choices=["both", "ollama", "vllm"])
    parser.add_argument("--levels", default="1,2,4,8",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    results = run(backend=args.backend, levels=levels,
                  max_tokens=args.max_tokens)

    print_table(results)
    # Also save JSON
    out = Path(__file__).parent.parent / "results" / "concurrency_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {out}", file=sys.stderr)
