"""Concurrency throughput benchmark.

Sends N simultaneous requests to server backends and measures aggregate
tokens/sec at each concurrency level.

Backends:
  ollama    — Ollama HTTP API (localhost:11434)
  mlx       — mlx_lm.server OpenAI-compat (localhost:8081)
  llamacpp  — llama-server OpenAI-compat (localhost:8082)
  vllm      — vLLM Metal HTTP (localhost:8765)

Usage:
    uv run python backends/bench_concurrency.py
    uv run python backends/bench_concurrency.py --backends ollama,mlx --levels 1,2,4
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
INPUT_LENGTHS = [512, 2048, 8192]
MAX_CTX = 32768

OLLAMA_URL  = "http://localhost:11434/api/generate"
MLX_URL     = "http://localhost:8081/v1/completions"
MLX_MODEL   = "/Users/jaesolshin/Documents/GitHub/test_mlx/models/qwen3.5-9b-mlx-4bit"
LLAMA_URL   = "http://localhost:8082/v1/completions"
LLAMA_MODEL = "Qwen3.5-9B-Q4_K_M.gguf"
VLLM_URL    = "http://localhost:8765/v1/completions"
VLLM_MODEL  = "Qwen/Qwen3.5-9B"
OMLX_MODELS_URL = "http://localhost:8000/v1/models"
OMLX_CHAT_URL   = "http://localhost:8000/v1/chat/completions"

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


# ── Ollama ────────────────────────────────────────────────────────────────────

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


# ── OpenAI-compat (mlx_lm.server / llama-server / vLLM) ─────────────────────

def _openai_compat_request(url: str, model: str,
                            prompt: str, max_tokens: int) -> RequestResult:
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    try:
        with requests.post(
            url,
            json={"model": model, "prompt": prompt,
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
                    if chunk["choices"][0].get("text"):
                        token_count += 1
        end = time.perf_counter()
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        return RequestResult(tokens=token_count, latency_sec=end - start,
                             ttft_ms=ttft, success=True)
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(tokens=0, latency_sec=end - start,
                             ttft_ms=0, success=False, error=str(e))


def _mlx_request(prompt: str, max_tokens: int) -> RequestResult:
    return _openai_compat_request(MLX_URL, MLX_MODEL, prompt, max_tokens)

def _llamacpp_request(prompt: str, max_tokens: int) -> RequestResult:
    return _openai_compat_request(LLAMA_URL, LLAMA_MODEL, prompt, max_tokens)

def _vllm_request(prompt: str, max_tokens: int) -> RequestResult:
    return _openai_compat_request(VLLM_URL, VLLM_MODEL, prompt, max_tokens)


# ── omlx (chat completions, SSE) ──────────────────────────────────────────────

_omlx_model: str | None = None


def _omlx_request(prompt: str, max_tokens: int) -> RequestResult:
    global _omlx_model
    if _omlx_model is None:
        resp = requests.get(OMLX_MODELS_URL, timeout=10)
        resp.raise_for_status()
        model = resp.json()["data"][0]["id"]
        _omlx_model = model
    else:
        model = _omlx_model

    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    try:
        with requests.post(
            OMLX_CHAT_URL,
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
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
                    _delta = chunk["choices"][0]["delta"]
                    if _delta.get("content") or _delta.get("reasoning_content"):
                        token_count += 1
        end = time.perf_counter()
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        return RequestResult(tokens=token_count, latency_sec=end - start,
                             ttft_ms=ttft, success=True)
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(tokens=0, latency_sec=end - start,
                             ttft_ms=0, success=False, error=str(e))


# ── Server health checks ──────────────────────────────────────────────────────

_HEALTH = {
    "ollama":   ("http://localhost:11434", False),
    "mlx":      ("http://localhost:8081/health", True),
    "llamacpp": ("http://localhost:8082/health", True),
    "vllm":     ("http://localhost:8765/health", True),
    "omlx":     ("http://localhost:8000/v1/models", True),
}

_FNS = {
    "ollama":   _ollama_request,
    "mlx":      _mlx_request,
    "llamacpp": _llamacpp_request,
    "vllm":     _vllm_request,
    "omlx":     _omlx_request,
}


def _check_server(backend: str) -> bool:
    url, needs_200 = _HEALTH[backend]
    try:
        r = requests.get(url, timeout=3)
        return r.status_code == 200 if needs_200 else True
    except Exception:
        return False


# ── Runner ────────────────────────────────────────────────────────────────────

def run_concurrency_level(backend: str, n: int, prompt: str, max_tokens: int) -> dict:
    fn = _FNS[backend]
    wall_start = time.perf_counter()
    results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(fn, prompt, max_tokens) for _ in range(n)]
        for f in as_completed(futures):
            results.append(f.result())

    wall_sec = time.perf_counter() - wall_start
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


def run(backends: list[str] = None, levels: list[int] = None,
        input_lengths: list[int] = None, max_tokens: int = MAX_TOKENS) -> list[dict]:
    if backends is None:
        backends = list(_FNS.keys())
    if levels is None:
        levels = CONCURRENCY_LEVELS
    if input_lengths is None:
        input_lengths = INPUT_LENGTHS

    active = []
    for b in backends:
        if _check_server(b):
            active.append(b)
        else:
            print(f"[WARN] {b} not running — skipping", file=sys.stderr)

    results = []
    for b in active:
        fn = _FNS[b]
        print(f"\n=== {b.upper()} ===", file=sys.stderr)
        for length in input_lengths:
            prompt = make_prompt(length)
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
    parser.add_argument("--backends", default="ollama,mlx",
                        help="Comma-separated: ollama,mlx,llamacpp,vllm")
    parser.add_argument("--levels", default="1,2,4",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--lengths", default="512,2048,8192",
                        help="Comma-separated input token lengths")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    backends = args.backends.split(",")
    levels = [int(x) for x in args.levels.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    results = run(backends=backends, levels=levels, input_lengths=lengths,
                  max_tokens=args.max_tokens)

    print_table(results)
    out = Path(__file__).parent.parent / "results" / "concurrency_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {out}", file=sys.stderr)
