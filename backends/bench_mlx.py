"""MLX backend benchmark for Qwen3.5-9B (4-bit quantized)."""

import time
from pathlib import Path

MLX_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "qwen3.5-9b-mlx-4bit")
HF_MODEL_ID = "Qwen/Qwen3.5-9B"


def run(prompt: str, max_tokens: int = 200) -> dict:
    from mlx_lm import load, stream_generate

    model_path = MLX_MODEL_PATH if Path(MLX_MODEL_PATH).exists() else HF_MODEL_ID

    model, tokenizer = load(model_path)

    # warm-up (1 token) to exclude model-load latency from timing
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=1):
        break

    tokens = []
    first_token_time = None
    start = time.perf_counter()

    for token in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens.append(token)

    end = time.perf_counter()

    total_tokens = len(tokens)
    total_sec = end - start
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0

    return {
        "backend": "MLX (4-bit)",
        "model": model_path,
        "tokens_per_sec": total_tokens / total_sec if total_sec > 0 else 0,
        "ttft_ms": ttft_ms,
        "total_tokens": total_tokens,
        "total_sec": total_sec,
    }


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PROMPT, MAX_TOKENS
    result = run(PROMPT, MAX_TOKENS)
    print(json.dumps(result, indent=2))
