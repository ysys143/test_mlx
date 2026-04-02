# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # Install all dependencies (Python 3.12)

# Run benchmarks
uv run python benchmark.py           # All backends (MLX, Ollama, llama.cpp, vLLM)
uv run python backends/bench_mlx.py  # Single backend
uv run python backends/bench_prefill_decode.py --backends mlx,ollama,llamacpp
uv run python backends/bench_concurrency.py --backend ollama --levels 1,2,4

# Visualization
uv run python plot_results.py        # Regenerate figures from results/

# Lint / test
ruff check                           # Lint
pytest                               # Tests
```

## Architecture

**Plugin-based benchmark system.** Each file in `backends/` is an independent runner implementing a consistent interface: a `run(prompt, max_tokens)` function returning `{"tokens_per_sec": float, "ttft_ms": float, "total_sec": float}`.

**Orchestration flow:**
1. `benchmark.py` — imports backend runners, executes N warm-up + timed runs, aggregates mean/stdev, writes `results/` JSON + Markdown
2. `config.py` — single source of truth for `PROMPT`, `MAX_TOKENS=200`, `RUNS=3`
3. `plot_results.py` — reads `results/*.json`, produces dark-theme Matplotlib figures into `figures/`

**Two specialized benchmark dimensions:**
- `bench_prefill_decode.py` — separates TTFT (prefill/KV-fill) from decode throughput across 64/512/2048 token inputs
- `bench_concurrency.py` — ThreadPoolExecutor parallel requests to test `OLLAMA_NUM_PARALLEL` scaling

**Backend connectivity modes:**
- MLX: direct library via `mlx_lm.load()` + `stream_generate()`
- Ollama / llama-server / mlx_lm.server: HTTP JSON (localhost ports)
- vLLM: both direct `vllm.LLM()` and OpenAI-compat HTTP API

**Model assets** live in `models/` (not committed):
- `qwen3.5-9b-mlx-4bit/` — MLX 4-bit quantized weights
- `Qwen3.5-9B-Q4_K_M.gguf` — llama.cpp GGUF

**Key finding encoded in design:** Qwen3.5's GatedDeltaNet linear attention achieves near-O(n) prefill but breaks true batching — so `bench_concurrency.py` intentionally tests aggregate throughput rather than per-request batch efficiency.
