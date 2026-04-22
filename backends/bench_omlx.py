"""omlx backend benchmark for Qwen3.5-9B.

omlx is an OpenAI-compatible inference server for Apple Silicon with
continuous batching and SSD KV caching (https://github.com/jundot/omlx).

Requires: omlx running at localhost:8000
    omlx serve --model-dir ~/models
"""

import json
import time

import requests

PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
MODELS_URL = f"{BASE_URL}/v1/models"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"

_cached_model: str | None = None


def _get_model() -> str:
    global _cached_model
    if _cached_model is None:
        resp = requests.get(MODELS_URL, timeout=10)
        resp.raise_for_status()
        model = resp.json()["data"][0]["id"]
        _cached_model = model
        return model
    return _cached_model


def run(prompt: str, max_tokens: int = 200) -> dict:
    model = _get_model()
    messages = [{"role": "user", "content": prompt}]

    # warm-up (non-streaming to avoid SSE parse overhead)
    requests.post(
        CHAT_URL,
        json={"model": model, "messages": messages, "max_tokens": 1, "stream": False},
        timeout=60,
    )

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    first_token_time = None
    token_count = 0
    wall_start = time.perf_counter()

    with requests.post(CHAT_URL, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                # Qwen3.5 reasoning models emit tokens in `reasoning_content`
                if delta.get("content") or delta.get("reasoning_content"):
                    token_count += 1

    wall_end = time.perf_counter()
    total_sec = wall_end - wall_start
    ttft_ms = (first_token_time - wall_start) * 1000 if first_token_time else 0

    return {
        "backend": "omlx",
        "model": model,
        "tokens_per_sec": token_count / total_sec if total_sec > 0 else 0,
        "ttft_ms": ttft_ms,
        "total_tokens": token_count,
        "total_sec": total_sec,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PROMPT, MAX_TOKENS
    result = run(PROMPT, MAX_TOKENS)
    print(json.dumps(result, indent=2))
