"""Prefill vs Decode breakdown benchmark.

Measures prefill time (TTFT) and decode throughput separately
across input lengths: 64, 512, 2048 tokens.

Backends: MLX, Ollama, llama.cpp, vLLM Metal (if running)

Usage:
    uv run python backends/bench_prefill_decode.py
    uv run python backends/bench_prefill_decode.py --backends mlx,ollama
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

# Prompt lengths to test (in tokens, approximate)
INPUT_LENGTHS = [64, 512, 2048, 8192, 32768]
DECODE_TOKENS = 200  # fixed output length
MAX_CTX = 32768  # must match across all backends

OLLAMA_URL = "http://localhost:11434/api/generate"
VLLM_URL = "http://localhost:8765/v1/completions"
VLLM_MODEL = "Qwen/Qwen3.5-9B"
OMLX_MODELS_URL = "http://localhost:8000/v1/models"
OMLX_CHAT_URL = "http://localhost:8000/v1/chat/completions"

# Base text repeated to fill token budget (~1 token per 4 chars for English)
_FILLER = (
    "The theory of relativity, developed by Albert Einstein, fundamentally changed "
    "our understanding of space, time, and gravity. "
)


def make_prompt(target_tokens: int) -> str:
    """Build a prompt of approximately target_tokens tokens."""
    chars_needed = target_tokens * 4
    text = (_FILLER * (chars_needed // len(_FILLER) + 1))[:chars_needed]
    return text + " Summarize the above passage concisely."


# ── MLX ──────────────────────────────────────────────────────────────────────

def _bench_mlx(prompt: str, decode_tokens: int) -> dict:
    from mlx_lm import load, stream_generate
    import mlx.core as mx

    model_path = str(Path(__file__).parent.parent / "models" / "qwen3.5-9b-mlx-4bit")
    model, tokenizer = load(model_path)

    # warm-up
    for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=1):
        break
    mx.eval()

    first_token_time = None
    token_times = []
    start = time.perf_counter()

    for response in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=decode_tokens):
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        if response.text:
            token_times.append(now)

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    decode_tokens_actual = len(token_times)
    decode_sec = (end - first_token_time) if first_token_time else 0
    decode_tps = decode_tokens_actual / decode_sec if decode_sec > 0 else 0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "decode_tok_per_sec": round(decode_tps, 2),
        "decode_tokens": decode_tokens_actual,
    }


# ── Ollama ────────────────────────────────────────────────────────────────────

def _bench_ollama(prompt: str, decode_tokens: int) -> dict:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": "qwen3.5:9b", "prompt": prompt, "stream": False,
              "options": {"num_predict": decode_tokens, "num_ctx": MAX_CTX}},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    # Ollama reports nanoseconds
    prompt_eval_ns = data.get("prompt_eval_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    eval_count = data.get("eval_count", 0)

    ttft_ms = prompt_eval_ns / 1e6
    decode_tps = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "decode_tok_per_sec": round(decode_tps, 2),
        "decode_tokens": eval_count,
    }


# ── llama.cpp ─────────────────────────────────────────────────────────────────

def _bench_llamacpp(prompt: str, decode_tokens: int) -> dict:
    model_path = str(
        Path(__file__).parent.parent / "models" / "Qwen3.5-9B-Q4_K_M.gguf"
    )
    cmd = [
        "llama-cli", "-m", model_path,
        "-p", prompt,
        "--predict", str(decode_tokens),
        "-ngl", "99",
        "-c", str(MAX_CTX),
        "--single-turn",
    ]
    proc = subprocess.run(cmd, input="", capture_output=True, text=True, timeout=600)
    combined = proc.stderr + "\n" + proc.stdout

    # Parse: "prompt eval time = X ms / N tokens"
    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", combined)
    ttft_ms = float(m.group(1)) if m else 0.0

    # Parse: "eval time = X ms / N tokens"
    m2 = re.search(r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", combined)
    if m2:
        eval_ms = float(m2.group(1))
        eval_tokens = int(m2.group(2))
        decode_tps = eval_tokens / (eval_ms / 1000) if eval_ms > 0 else 0
    else:
        eval_tokens, decode_tps = 0, 0.0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "decode_tok_per_sec": round(decode_tps, 2),
        "decode_tokens": eval_tokens,
    }


# ── vLLM Metal ────────────────────────────────────────────────────────────────

def _bench_vllm(prompt: str, decode_tokens: int) -> dict:
    first_token_time = None
    token_count = 0
    start = time.perf_counter()

    with requests.post(
        VLLM_URL,
        json={"model": VLLM_MODEL, "prompt": prompt,
              "max_tokens": decode_tokens, "stream": True},
        stream=True, timeout=600,
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
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    decode_sec = (end - first_token_time) if first_token_time else 0
    decode_tps = token_count / decode_sec if decode_sec > 0 else 0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "decode_tok_per_sec": round(decode_tps, 2),
        "decode_tokens": token_count,
    }


# ── omlx ──────────────────────────────────────────────────────────────────────

_omlx_model: str | None = None


def _bench_omlx(prompt: str, decode_tokens: int) -> dict:
    global _omlx_model
    if _omlx_model is None:
        resp = requests.get(OMLX_MODELS_URL, timeout=10)
        resp.raise_for_status()
        model = resp.json()["data"][0]["id"]
        _omlx_model = model
    else:
        model = _omlx_model

    messages = [{"role": "user", "content": prompt}]
    first_token_time = None
    token_count = 0
    start = time.perf_counter()

    with requests.post(
        OMLX_CHAT_URL,
        json={"model": model, "messages": messages,
              "max_tokens": decode_tokens, "stream": True},
        stream=True, timeout=600,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                chunk = json.loads(line[6:])
                _delta = chunk["choices"][0]["delta"]
                if _delta.get("content") or _delta.get("reasoning_content"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    decode_sec = (end - first_token_time) if first_token_time else 0
    decode_tps = token_count / decode_sec if decode_sec > 0 else 0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "decode_tok_per_sec": round(decode_tps, 2),
        "decode_tokens": token_count,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

BACKEND_FNS = {
    "mlx": _bench_mlx,
    "ollama": _bench_ollama,
    "llamacpp": _bench_llamacpp,
    "vllm": _bench_vllm,
    "omlx": _bench_omlx,
}

BACKEND_LABELS = {
    "mlx": "MLX 4-bit",
    "ollama": "Ollama",
    "llamacpp": "llama.cpp Metal",
    "vllm": "vLLM Metal (HTTP)",
    "omlx": "omlx",
}


def run(backends: list[str] = None, input_lengths: list[int] = None,
        decode_tokens: int = DECODE_TOKENS) -> list[dict]:
    if backends is None:
        backends = ["mlx", "ollama", "llamacpp"]
    if input_lengths is None:
        input_lengths = INPUT_LENGTHS

    # Check vLLM availability
    if "vllm" in backends:
        try:
            requests.get("http://localhost:8765/health", timeout=2)
        except Exception:
            print("[WARN] vLLM Metal not running — skipping", file=sys.stderr)
            backends = [b for b in backends if b != "vllm"]

    results = []
    for length in input_lengths:
        prompt = make_prompt(length)
        print(f"\n=== Input ~{length} tokens ===", file=sys.stderr)

        for backend in backends:
            print(f"  {BACKEND_LABELS[backend]}...", file=sys.stderr)
            try:
                r = BACKEND_FNS[backend](prompt, decode_tokens)
                row = {
                    "backend": BACKEND_LABELS[backend],
                    "input_tokens": length,
                    **r,
                }
                results.append(row)
                print(f"    TTFT: {r['ttft_ms']}ms | decode: {r['decode_tok_per_sec']} tok/s",
                      file=sys.stderr)
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)

    return results


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Backend':<20} {'Input tok':>10} {'TTFT (ms)':>10} {'Decode tok/s':>13}")
    print("-" * 70)

    prev_len = None
    for r in results:
        if r["input_tokens"] != prev_len:
            if prev_len is not None:
                print()
            prev_len = r["input_tokens"]
        print(f"{r['backend']:<20} {r['input_tokens']:>10} "
              f"{r['ttft_ms']:>10.1f} {r['decode_tok_per_sec']:>13.2f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="mlx,ollama,llamacpp",
                        help="Comma-separated: mlx,ollama,llamacpp,vllm")
    parser.add_argument("--lengths", default="64,512,2048",
                        help="Comma-separated input token lengths")
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    args = parser.parse_args()

    backends = args.backends.split(",")
    lengths = [int(x) for x in args.lengths.split(",")]

    results = run(backends=backends, input_lengths=lengths,
                  decode_tokens=args.decode_tokens)
    print_table(results)

    out = Path(__file__).parent.parent / "results" / "prefill_decode_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {out}", file=sys.stderr)
