"""llama.cpp (Metal) backend benchmark for Qwen3.5-9B Q4_K_M."""

import re
import subprocess
import time
from pathlib import Path

LLAMA_CLI = "llama-cli"
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "Qwen3.5-9B-Q4_K_M.gguf")
N_GPU_LAYERS = 99  # offload all layers to Metal


def _parse_perf(stderr: str) -> dict:
    """Parse timing info from llama-cli stderr output."""
    patterns = {
        "prompt_tokens": r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens",
        "prompt_ms": r"prompt eval time\s*=\s*([\d.]+)\s*ms",
        "eval_tokens": r"\beval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens",
        "eval_ms": r"\beval time\s*=\s*([\d.]+)\s*ms",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, stderr)
        result[key] = float(m.group(1)) if m else 0.0
    return result


def run(prompt: str, max_tokens: int = 200) -> dict:
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--predict", str(max_tokens),
        "-ngl", str(N_GPU_LAYERS),
        "--log-disable",   # suppress verbose logs to stdout
        "-no-cnv",         # no conversation mode
    ]

    wall_start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall_end = time.perf_counter()

    perf = _parse_perf(proc.stderr)

    eval_tokens = int(perf["eval_tokens"])
    eval_ms = perf["eval_ms"]
    prompt_ms = perf["prompt_ms"]

    tokens_per_sec = (eval_tokens / (eval_ms / 1000)) if eval_ms > 0 else 0

    return {
        "backend": "llama.cpp Metal",
        "model": MODEL_PATH,
        "tokens_per_sec": tokens_per_sec,
        "ttft_ms": prompt_ms,
        "total_tokens": eval_tokens,
        "total_sec": wall_end - wall_start,
    }


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PROMPT, MAX_TOKENS
    result = run(PROMPT, MAX_TOKENS)
    print(json.dumps(result, indent=2))
