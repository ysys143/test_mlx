"""llama.cpp (Metal) backend benchmark for Qwen3.5-9B Q4_K_M."""

import re
import subprocess
import time
from pathlib import Path

LLAMA_CLI = "llama-cli"
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "Qwen3.5-9B-Q4_K_M.gguf")
N_GPU_LAYERS = 99  # offload all layers to Metal


def _parse_perf(stderr: str, stdout: str) -> dict:
    """Parse timing info from llama-cli output.

    Tries two formats:
    1. llama_perf_context_print (classic format)
    2. Interactive display: [ Prompt: X t/s | Generation: Y t/s ]
    """
    # Format 1: classic llama_perf_context_print
    patterns = {
        "prompt_tokens": r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens",
        "prompt_ms": r"prompt eval time\s*=\s*([\d.]+)\s*ms",
        "eval_tokens": r"\beval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens",
        "eval_ms": r"\beval time\s*=\s*([\d.]+)\s*ms",
    }
    result = {}
    combined = stderr + "\n" + stdout
    for key, pat in patterns.items():
        m = re.search(pat, combined)
        result[key] = float(m.group(1)) if m else 0.0

    # Format 2: interactive display [ Prompt: X t/s | Generation: Y t/s ]
    if result["eval_ms"] == 0:
        m = re.search(r"Generation:\s*([\d.]+)\s*t/s", combined)
        if m:
            gen_tps = float(m.group(1))
            # estimate ms from token count and rate
            result["_gen_tps"] = gen_tps
        m2 = re.search(r"Prompt:\s*([\d.]+)\s*t/s", combined)
        if m2:
            result["_prompt_tps"] = float(m2.group(1))

    return result


def run(prompt: str, max_tokens: int = 200) -> dict:
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--predict", str(max_tokens),
        "-ngl", str(N_GPU_LAYERS),
        "--single-turn",   # exit after first response (no interactive wait)
    ]

    wall_start = time.perf_counter()
    # input="" closes stdin immediately so conversation mode doesn't block
    proc = subprocess.run(cmd, input="", capture_output=True, text=True, timeout=1200)
    wall_end = time.perf_counter()
    total_sec = wall_end - wall_start

    perf = _parse_perf(proc.stderr, proc.stdout)

    # Prefer classic llama_perf format; fall back to interactive display
    if perf["eval_ms"] > 0:
        eval_tokens = int(perf["eval_tokens"])
        tokens_per_sec = eval_tokens / (perf["eval_ms"] / 1000)
        prompt_ms = perf["prompt_ms"]
    elif "_gen_tps" in perf:
        tokens_per_sec = perf["_gen_tps"]
        eval_tokens = max_tokens
        prompt_ms = (1000 / perf.get("_prompt_tps", 1)) if "_prompt_tps" in perf else 0
    else:
        tokens_per_sec = max_tokens / total_sec if total_sec > 0 else 0
        eval_tokens = max_tokens
        prompt_ms = 0

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
