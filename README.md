# Apple Silicon LLM Inference Benchmark

Benchmarks for Qwen3.5-9B inference across MLX, llama.cpp, Ollama, and vLLM Metal on Apple Silicon.

## Single-Request Throughput

```
tok/s  (higher is better)                               512-tok input, 200-tok output

MLX 4-bit       ████████████████████████████████████  34.52
llama.cpp Metal ███████████████████████████████████   33.10
Ollama          ███████████████████████████████       29.99
vLLM direct     ████████████████                      15.61
vLLM HTTP       ███                                    3.24
                0         10        20        30        35
```

```
TTFT ms  (lower is better)

Ollama          ████████████                          126 ms
MLX 4-bit       █████████████████████████             256 ms
vLLM direct     ████████████████████████████████████  341 ms
llama.cpp Metal ████████████████████████████████████  344 ms
vLLM HTTP       █████████████████████████████████████ 366 ms
                0        100       200       300       366
```

## Prefill Time (TTFT) vs Input Length

```
TTFT ms  (lower is better)

            64 tok   512 tok   2048 tok   8192 tok   32768 tok
            ------   -------   --------   --------   ---------
MLX 4-bit     ~60      ~256      ~900      ~3500      ~18000
llama.cpp     ~80      ~344     ~1200      ~4500      ~22000
Ollama        ~50      ~126      ~500      ~2000      ~10000

TTFT growth is near-linear up to 8k (GatedDeltaNet O(n)).
At 32k, SDPA layers cause superlinear growth -- TTFT roughly 5x expected.
```

```
Decode tok/s vs input length  (higher is better)

            512 tok    2048 tok   8192 tok   32768 tok
            -------    --------   --------   ---------
MLX 4-bit    ~34         ~34        ~32        ~17      <- halved at 32k
llama.cpp    ~33         ~33        ~31        ~16
Ollama       ~30         ~30        ~28        ~14
```

Decode throughput is stable up to 8k, then roughly halves at 32k.
SDPA layers require O(n) KV cache reads per decode step.

## Concurrency (Ollama, OLLAMA_NUM_PARALLEL=4)

```
Aggregate tok/s vs concurrency  (ideal: linear scaling)

 512-tok input
 c=1  ████████████████████████  23.5   <- baseline
 c=2  █████████████████████████ 24.0   [X] no scaling (ideal: ~47)
 c=4  ██████████████████████████25.0   [X] no scaling (ideal: ~94)

2048-tok input
 c=1  █████████████████████████ 21.7
 c=2  █████████████████████████ 21.3   [X] no scaling
 c=4  █████████████████████████ 21.4   [X] no scaling

8192-tok input
 c=1  █████████████████████     18.1
 c=2  █████████████████████     18.2   [X] no scaling
 c=4  ████████████████████      17.2   [X] slight degradation
```

```
TTFT grows linearly with queue depth -- proof of serial processing

 512-tok c=1  ██                                1179 ms
 512-tok c=2  █████████████                     5500 ms  (4.7x)
 512-tok c=4  █████████████████████████████████13081 ms  (11.1x)
```

Aggregate tok/s is flat -- `NUM_PARALLEL` controls queue depth only.
Requests are processed sequentially; TTFT scales linearly with queue depth.

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
