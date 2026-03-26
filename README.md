# Apple Silicon LLM Inference Benchmark

Benchmarks for Qwen3.5-9B inference across MLX, llama.cpp, Ollama, and vLLM Metal on Apple Silicon.

## Single-Request Throughput

![Throughput and TTFT](figures/fig1_throughput.png)

## Prefill Time (TTFT) vs Input Length

![Prefill scaling](figures/fig2_prefill_scaling.png)

TTFT grows near-linearly up to 8k (GatedDeltaNet O(n) linear attention). At 32k, SDPA layers cause superlinear growth.

## Decode Throughput vs Input Length

![Decode throughput](figures/fig3_decode_vs_length.png)

Stable up to 8k, then roughly halved at 32k — SDPA layers require reading the full KV cache on every decode step.

## Concurrency (Ollama, OLLAMA_NUM_PARALLEL=4)

![Concurrency benchmark](figures/fig4_concurrency.png)

Aggregate tok/s is completely flat regardless of concurrency — `NUM_PARALLEL` controls queue depth only, not batching. TTFT scales linearly with queue depth (proof of serial GPU processing).

Full report: [`results/benchmark_report.md`](results/benchmark_report.md)

## Key Findings

- **MLX and llama.cpp Metal** are neck-and-neck (~34 vs ~33 tok/s)
- **Ollama** has the lowest TTFT (126ms) — persistent server keeps model warm
- **vLLM Metal HTTP** adds 4.8x overhead vs direct library call; bf16 vs 4-bit adds another 2.2x (~10x total vs MLX)
- **Qwen3.5 GatedDeltaNet** linear attention gives near-O(n) prefill scaling but breaks batching — no backend achieves true parallel request processing
- **OLLAMA_NUM_PARALLEL** does not enable batching; aggregate throughput never scales with concurrency

## Setup

```bash
uv sync
```

### Models

```bash
# MLX 4-bit
uv run mlx_lm.convert --model Qwen/Qwen3.5-9B -q --mlx-path ./models/qwen3.5-9b-mlx-4bit

# llama.cpp GGUF
hf download unsloth/Qwen3.5-9B-GGUF --include "Qwen3.5-9B-Q4_K_M.gguf" --local-dir ./models/

# Ollama
ollama pull qwen3.5:9b

# vLLM Metal
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Usage

```bash
# Single-request throughput (all backends)
uv run python benchmark.py

# Individual backends
uv run python backends/bench_mlx.py
uv run python backends/bench_ollama.py
uv run python backends/bench_llamacpp.py
uv run python backends/bench_vllm_metal.py
uv run python backends/bench_vllm_metal_direct.py

# Prefill vs decode breakdown by input length
uv run python backends/bench_prefill_decode.py --backends mlx,ollama,llamacpp

# Concurrency throughput
uv run python backends/bench_concurrency.py --backend ollama --levels 1,2,4
```
