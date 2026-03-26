"""vLLM Metal direct benchmark — no HTTP server.

Isolates HTTP/scheduler overhead from model computation by calling
mlx_lm.stream_generate directly (same as vLLM Metal does internally),
using the same HuggingFace full-precision model that vLLM Metal loads.

This tells us: is the 3.24 tok/s bottleneck from
  (a) the HTTP + vLLM engine overhead, or
  (b) the full-precision (bf16) model vs 4-bit quantized?
"""

import sys
import time

# Use vllm-metal venv's mlx_lm (may differ from project venv)
VLLM_METAL_SITE = (
    "/Users/jaesolshin/.venv-vllm-metal/lib/python3.12/site-packages"
)
if VLLM_METAL_SITE not in sys.path:
    sys.path.insert(0, VLLM_METAL_SITE)

# HF model name — same as what vLLM Metal serves
MODEL = "Qwen/Qwen3.5-9B"


def run(prompt: str, max_tokens: int = 200) -> dict:
    from mlx_lm import load, stream_generate
    import mlx.core as mx

    model, tokenizer = load(MODEL)

    # warm-up: 1 token
    for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=1):
        break
    mx.eval()

    token_count = 0
    first_token_time = None
    start = time.perf_counter()

    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if response.text:
            token_count += 1

    end = time.perf_counter()
    total_sec = end - start
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0

    return {
        "backend": "vLLM Metal (direct, HF bf16)",
        "model": MODEL,
        "tokens_per_sec": token_count / total_sec if total_sec > 0 else 0,
        "ttft_ms": ttft_ms,
        "total_tokens": token_count,
        "total_sec": total_sec,
    }


if __name__ == "__main__":
    import json
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from config import PROMPT, MAX_TOKENS
    result = run(PROMPT, MAX_TOKENS)
    print(json.dumps(result, indent=2))
