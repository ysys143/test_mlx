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

CONCURRENCY_LEVELS = [1, 2, 4]
INPUT_LENGTHS = [512, 2048, 8192]   # token counts for long-input tests
MAX_CTX = 32768

# Backend endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
VLLM_URL = "http://localhost:8765/v1/completions"
VLLM_MODEL = "Qwen/Qwen3.5-9B"

_FILLER = (
    "The theory of relativity, developed by Albert Einstein, fundamentally changed "
    "our understanding of space, time, and gravity. "
)


def make_prompt(target_tokens: int) -> str:
    chars_needed = target_tokens * 4
    text = (_FILLER * (chars_needed // len(_FILLER) + 1))[:chars_needed]
    return text + " Summarize the above passage concisely."


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
                  "options": {"num_predict": max_tokens, "num_ctx": MAX_CTX}},
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


def run(backend: str = "both", levels: list[int] = None,
        input_lengths: list[int] = None, max_tokens: int = MAX_TOKENS) -> list[dict]:
    if levels is None:
        levels = CONCURRENCY_LEVELS
    if input_lengths is None:
        input_lengths = INPUT_LENGTHS

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
            print("[WARN] vLLM Metal not running — skipping", file=sys.stderr)

    results = []
    for b in backends:
        fn = _ollama_request if b == "ollama" else _vllm_request
        print(f"\n=== {b.upper()} ===", file=sys.stderr)

        for length in input_lengths:
            prompt = make_prompt(length)
            # warm-up
            print(f"  [input ~{length} tok] warm-up...", file=sys.stderr)
            fn(prompt, 5)

            for n in levels:
                print(f"  [input ~{length} tok] concurrency={n}...", file=sys.stderr)
                r = run_concurrency_level(b, n, prompt, max_tokens)
                r["input_tokens"] = length
                results.append(r)
                print(f"    {r['aggregate_tok_per_sec']} tok/s agg, "
                      f"avg TTFT {r['avg_ttft_ms']}ms, "
                      f"avg lat {r['avg_latency_sec']}s", file=sys.stderr)

    return results


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 82)
    print(f"{'Backend':<10} {'Input tok':>10} {'Concurr':>8} {'Agg tok/s':>10} "
          f"{'Avg lat(s)':>11} {'Avg TTFT(ms)':>13}")
    print("-" * 82)
    prev = None
    for r in results:
        key = (r["backend"], r.get("input_tokens", 0))
        if key != prev:
            if prev is not None:
                print()
            prev = key
        print(f"{r['backend']:<10} {r.get('input_tokens', '-'):>10} "
              f"{r['concurrency']:>8} "
              f"{r['aggregate_tok_per_sec']:>10.1f} "
              f"{r['avg_latency_sec']:>11.2f} "
              f"{r['avg_ttft_ms']:>13.0f}")
    print("=" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="both",
                        choices=["both", "ollama", "vllm"])
    parser.add_argument("--levels", default="1,2,4",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--lengths", default="512,2048,8192",
                        help="Comma-separated input token lengths")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    results = run(backend=args.backend, levels=levels, input_lengths=lengths,
                  max_tokens=args.max_tokens)

    print_table(results)
    # Also save JSON
    out = Path(__file__).parent.parent / "results" / "concurrency_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {out}", file=sys.stderr)
