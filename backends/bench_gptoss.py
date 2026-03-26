"""openharmony-mlx GPT-OSS-20B (4-bit) benchmark.

Uses /tmp/openharmony-mlx cloned repo directly (no pip install needed).
"""

import sys
import time
from pathlib import Path

# Add cloned repo to path (bypasses custom build backend requirement)
REPO_PATH = "/tmp/openharmony-mlx"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

MODEL_DIR = str(Path(__file__).parent.parent / "gpt-oss-20b" / "original")


def run(prompt: str, max_tokens: int = 200) -> dict:
    from gpt_oss.mlx_gpt_oss.generate import TokenGenerator

    generator = TokenGenerator(MODEL_DIR)

    prompt_tokens = generator.tokenizer.encode(prompt)

    # warm-up
    for _ in generator.generate(prompt_tokens, stop_tokens=[], max_tokens=1):
        break

    tokens = []
    first_token_time = None
    start = time.perf_counter()

    for token in generator.generate(prompt_tokens, stop_tokens=[], max_tokens=max_tokens):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens.append(token)

    end = time.perf_counter()

    total_tokens = len(tokens)
    total_sec = end - start
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0

    return {
        "backend": "openharmony-mlx (GPT-OSS-20B 4-bit)",
        "model": MODEL_DIR,
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
