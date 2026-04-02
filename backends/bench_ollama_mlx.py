"""Ollama MLX backend benchmark (requires Ollama >= 0.19 on Apple Silicon).

Ollama 0.19 replaced llama.cpp with MLX as the inference engine on Apple Silicon.
This backend uses the same HTTP API as bench_ollama.py but targets the MLX-optimised
NVFP4 model tag and labels results separately for side-by-side comparison.

Model setup:
    ollama pull qwen3.5:9b   # Ollama 0.19+ will pull the MLX/NVFP4 variant on Apple Silicon
"""

import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"  # Ollama 0.19+ serves this via MLX on Apple Silicon


def run(prompt: str, max_tokens: int = 200) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }

    # warm-up
    requests.post(OLLAMA_URL, json={**payload, "options": {"num_predict": 1}}, timeout=120)

    wall_start = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    wall_end = time.perf_counter()

    data = resp.json()

    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)

    tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0
    ttft_ms = prompt_eval_duration_ns / 1e6

    return {
        "backend": "Ollama (MLX)",
        "model": MODEL,
        "tokens_per_sec": tokens_per_sec,
        "ttft_ms": ttft_ms,
        "total_tokens": eval_count,
        "total_sec": wall_end - wall_start,
    }


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PROMPT, MAX_TOKENS
    result = run(PROMPT, MAX_TOKENS)
    print(json.dumps(result, indent=2))
