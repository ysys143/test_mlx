# Apple Silicon LLM Inference Benchmark

Benchmarks for Qwen3.5-9B inference speed across multiple backends on Apple Silicon (Metal GPU).

## Results

| Backend | tokens/sec | TTFT (ms) |
|---------|:----------:|:---------:|
| MLX 4-bit (mlx-lm 0.31.1) | **34.52** | 256 |
| llama.cpp Metal (Q4_K_M, -ngl 99) | **33.1** | ~344 |
| Ollama 0.18.2 | **29.99** | 126 |
| vLLM Metal 0.17.1 (direct, no HTTP) | **15.61** | 341 |
| vLLM Metal 0.17.1 (HTTP server) | **3.24** | 366 |

Full report: [`results/benchmark_report.md`](results/benchmark_report.md)

## Setup

```bash
uv sync
```

### Models

Download models manually before running:

```bash
# MLX 4-bit (Qwen3.5-9B)
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
# Run all backends
uv run python benchmark.py

# Run individual backends
uv run python backends/bench_mlx.py
uv run python backends/bench_ollama.py
uv run python backends/bench_llamacpp.py
uv run python backends/bench_vllm_metal.py
uv run python backends/bench_vllm_metal_direct.py
```

## Key Findings

- **MLX and llama.cpp Metal** are neck-and-neck (~34 vs ~33 tok/s) — both use Metal GPU directly
- **Ollama** has the lowest TTFT (126ms) due to persistent server keeping model warm
- **vLLM Metal HTTP server** adds 4.8x overhead vs direct library call for single-request inference
- **vLLM Metal paged attention** does not support Qwen3.5's GatedDeltaNet linear attention architecture
