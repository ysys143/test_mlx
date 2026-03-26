"""vLLM Metal backend benchmark for Qwen3.5-9B.

Tests two modes:
  - MLX path (default, paged attention disabled)
  - Paged attention (VLLM_METAL_USE_PAGED_ATTENTION=1)
"""

import json
import os
import signal
import subprocess
import time
import requests

VLLM_METAL_PYTHON = os.path.expanduser("~/.venv-vllm-metal/bin/python")
VLLM_METAL_BIN = os.path.expanduser("~/.venv-vllm-metal/bin/vllm")
MODEL = "Qwen/Qwen3.5-9B"
PORT = 8765
BASE_URL = f"http://localhost:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"
COMPLETIONS_URL = f"{BASE_URL}/v1/completions"


def _wait_for_server(timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False


def _run_single(prompt: str, max_tokens: int, paged: bool) -> dict:
    env = os.environ.copy()
    env["VLLM_METAL_USE_PAGED_ATTENTION"] = "1" if paged else "0"
    if not paged:
        env["VLLM_METAL_MEMORY_FRACTION"] = "auto"

    server_cmd = [
        VLLM_METAL_BIN, "serve", MODEL,
        "--port", str(PORT),
        "--max-model-len", "4096",
    ]

    proc = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not _wait_for_server():
            raise RuntimeError("vLLM Metal server failed to start")

        # warm-up
        requests.post(COMPLETIONS_URL, json={
            "model": MODEL, "prompt": prompt, "max_tokens": 1
        }, timeout=60)

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        }

        first_token_time = None
        token_count = 0
        wall_start = time.perf_counter()

        with requests.post(COMPLETIONS_URL, json=payload, stream=True, timeout=600) as resp:
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

        wall_end = time.perf_counter()
        total_sec = wall_end - wall_start
        ttft_ms = (first_token_time - wall_start) * 1000 if first_token_time else 0

        return {
            "backend": f"vLLM Metal ({'Paged' if paged else 'MLX path'})",
            "model": MODEL,
            "tokens_per_sec": token_count / total_sec if total_sec > 0 else 0,
            "ttft_ms": ttft_ms,
            "total_tokens": token_count,
            "total_sec": total_sec,
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)


def run(prompt: str, max_tokens: int = 200) -> list[dict]:
    """Returns results for both MLX path and paged attention modes."""
    results = []
    for paged in (False, True):
        results.append(_run_single(prompt, max_tokens, paged=paged))
        time.sleep(3)  # cooldown between runs
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PROMPT, MAX_TOKENS
    for r in run(PROMPT, MAX_TOKENS):
        print(json.dumps(r, indent=2))
