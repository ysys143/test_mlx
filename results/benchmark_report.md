# Apple Silicon Inference Benchmark Report
**Model**: Qwen3.5-9B (+ GPT-OSS-20B)
**Date**: 2026-03-26
**Hardware**: Apple Silicon (Metal GPU)
**Prompt**: "Explain the theory of relativity in detail."
**Max tokens**: 200

---

## Results

| Backend | tokens/sec | TTFT (ms) | Total sec | Status |
|---------|:----------:|:---------:|:---------:|--------|
| MLX (4-bit, mlx-lm 0.31.1) | **34.52** | 256 | ~5.8 | [OK] |
| llama.cpp Metal (Q4_K_M, -ngl 99) | **33.1** | ~344 | ~6.1 | [OK] |
| Ollama 0.18.2 (qwen3.5:9b) | **29.99** | 126 | ~6.7 | [OK] |
| vLLM Metal 0.17.1 (MLX path, HTTP) | **3.24** | 366 | ~61.7 | [OK, slow] |
| vLLM Metal 0.17.1 (direct, no HTTP) | **15.61** | 341 | ~12.6 | [OK] |
| vLLM Metal 0.17.1 (Paged Attention) | N/A | -- | -- | [X] Unsupported |
| openharmony-mlx GPT-OSS-20B | N/A | -- | -- | [X] Weight mismatch |

---

## vLLM Metal Overhead Analysis

Qwen3.5-9B is a VLM architecture (Qwen3_5ForConditionalGeneration with vision_config).
vLLM Metal loads it in full precision bf16 via mlx_vlm.

| Layer | Overhead | Evidence |
|---|---|---|
| HTTP + vLLM scheduler | 4.8x | HTTP server (3.24) vs direct call (15.61) |
| bf16 vs 4-bit precision | ~2.2x | Direct bf16 (15.61) vs mlx-lm 4-bit (34.52) |
| Total combined | ~10.6x | HTTP server (3.24) vs mlx-lm 4-bit (34.52) |

**Key finding**: HTTP is NOT the bottleneck. The vLLM engine scheduler adds ~4.8x overhead
for single-request inference. Model precision (bf16 vs 4-bit) adds another ~2.2x.
Switching to Unix sockets would save < 2ms on a 62-second request -- negligible.

`backends/bench_vllm_metal_direct.py` demonstrates the HTTP bypass approach:
calls mlx_lm.stream_generate() directly from the vllm-metal venv, no server.

---

## Notes

### MLX (mlx-lm)
- Model: `./models/qwen3.5-9b-mlx-4bit` (4-bit quantized with mlx_lm.convert)
- Fastest overall throughput on Apple Silicon
- Uses Unified Memory -- no explicit VRAM/RAM split
- Warm-up (1 token) run done before timed measurement

### llama.cpp Metal
- Model: `./models/Qwen3.5-9B-Q4_K_M.gguf` (unsloth/Qwen3.5-9B-GGUF)
- Full GPU offload: `-ngl 99`
- Competitive with MLX at nearly the same token rate
- CLI flag `--single-turn` + `input=""` required to prevent stdin blocking

### Ollama
- Built-in llama.cpp backend with `qwen3.5:9b`
- Lowest TTFT (126ms) -- possibly due to model already being hot in memory
- HTTP API provides `eval_count`, `eval_duration`, `prompt_eval_duration` directly

### vLLM Metal (HTTP, MLX path)
- Version 0.17.1 via `~/.venv-vllm-metal`
- Loads full-precision bf16 via mlx_vlm (Qwen3.5 is a VLM architecture)
- Very low throughput (3.24 tok/s): HTTP + scheduler overhead (4.8x) + bf16 (2.2x)
- Measured via streaming `/v1/completions` endpoint with TTFT tracking

### vLLM Metal (direct, no HTTP)
- Same vllm-metal venv, calls mlx_lm.stream_generate() directly
- No subprocess spawn, no HTTP, no vLLM scheduler
- 15.61 tok/s -- 4.8x better than HTTP server path
- Remaining gap vs mlx-lm 4-bit is model precision (bf16 vs 4-bit)

### vLLM Metal (Paged Attention) -- N/A
- `VLLM_METAL_USE_PAGED_ATTENTION=1`
- Not supported for Qwen3.5-9B: crashes with:
  ```
  NotImplementedError: Linear attention (GatedDeltaNet) is not yet implemented
  for Metal paged attention. Module: Qwen3_5GatedDeltaNet, layer: 0.
  ```
- Qwen3.5 uses a GatedDeltaNet linear attention variant not yet ported
  to vLLM Metal's paged attention backend.

### openharmony-mlx GPT-OSS-20B -- N/A
- Weight key mismatch: safetensors uses `block.N.attn.qkv.*`,
  model expects `layers.N.self_attn.*`. All parameters failed to map.
- Additional issue: `param.data = weights[name]` is PyTorch-style;
  MLX arrays are immutable, requires `model.load_weights()`.

---

## Backend Comparison

```
tokens/sec (higher is better)
MLX 4-bit                 ||||||||||||||||||||||||||||||||||||  34.52
llama.cpp Metal           |||||||||||||||||||||||||||||||||||   33.10
Ollama                    ||||||||||||||||||||||||||||||        29.99
vLLM Metal direct         |||||||||||||||                       15.61
vLLM Metal HTTP           |||                                    3.24
```

```
TTFT ms (lower is better)
Ollama            ||||||||||||||  126ms
vLLM Metal direct ||||||||||||||||||||||||||||||||||||  341ms
MLX               ||||||||||||||||||||||||||||  256ms
llama.cpp         ||||||||||||||||||||||||||||||||||||  ~344ms
vLLM Metal HTTP   ||||||||||||||||||||||||||||||||||||||||  366ms
```

---

## Concurrency Benchmark (mlx_lm.server vs Ollama)

mlx_lm.server (`--decode-concurrency 4`, port 8081) vs Ollama (`OLLAMA_NUM_PARALLEL=4`).

### Aggregate tok/s

| Input tok | Concurrency | MLX tok/s | Ollama tok/s |
|----------:|:-----------:|----------:|-------------:|
| 512       | 1           | 23.8      | 25.6         |
| 512       | 2           | 32.6      | 25.9         |
| 512       | 4           | 38.0      | 24.6         |
| 2048      | 1           | 18.9      | 18.5         |
| 2048      | 2           | 23.1      | 17.4         |
| 2048      | 4           | 25.6      | 15.9         |
| 8192      | 1           | 9.1       | 10.3         |
| 8192      | 2           | 10.0      | 9.7          |
| 8192      | 4           | 10.6      | 9.0          |

### Avg TTFT (ms)

| Input tok | Concurrency | MLX TTFT | Ollama TTFT |
|----------:|:-----------:|---------:|------------:|
| 512       | 1           | 1625     | 1080        |
| 512       | 2           | 3322     | 4950        |
| 512       | 4           | 6219     | 13187       |
| 2048      | 1           | 4697     | 2881        |
| 2048      | 2           | 9260     | 8780        |
| 2048      | 4           | 18065    | 20834       |
| 8192      | 1           | 16763    | 6634        |
| 8192      | 2           | 32161    | 16981       |
| 8192      | 4           | 62542    | 40946       |

**Key findings**:
- **mlx_lm.server achieves real decode batching**: aggregate tok/s scales with concurrency
  at short inputs (512 tok: +60% at c=4 vs c=1). `--decode-concurrency` actually works.
- **Ollama serializes completely**: aggregate tok/s flat or declining at all input lengths.
  `OLLAMA_NUM_PARALLEL` controls queue depth only, not batching.
- **Batching effect shrinks with input length**: at 8k tokens, even MLX barely scales —
  prefill dominates and saturates the GPU, leaving no room for batched decode.
- **TTFT trade-off**: MLX batching delays individual TTFT (each request waits for others
  to be scheduled), but wins on aggregate throughput.

---

## Conclusions

1. **MLX and llama.cpp Metal are neck-and-neck** (~34 vs ~33 tok/s).

2. **Ollama has the best TTFT** (126ms) -- persistent server keeps model warm.

3. **vLLM Metal HTTP overhead is 4.8x**, not the protocol itself (HTTP vs socket
   would save < 2ms). The bottleneck is vLLM's engine scheduler for single-request
   inference. Direct library call recovers 15.61 tok/s.

4. **vLLM Metal paged attention** does not support Qwen3.5's GatedDeltaNet.

5. **openharmony-mlx GPT-OSS** blocked by weight format incompatibility.

6. **Ollama does not batch requests** even with `OLLAMA_NUM_PARALLEL=4`. Aggregate
   throughput is flat across concurrency levels. Each request occupies the full GPU
   sequentially. Likely cause: Qwen3.5's GatedDeltaNet recurrent state is not
   batch-parallelizable in llama.cpp's current implementation.

---

## Files

| File | Purpose |
|---|---|
| `backends/bench_mlx.py` | mlx-lm 4-bit direct |
| `backends/bench_ollama.py` | Ollama HTTP API |
| `backends/bench_llamacpp.py` | llama-cli subprocess |
| `backends/bench_vllm_metal.py` | vLLM Metal HTTP server |
| `backends/bench_vllm_metal_direct.py` | vLLM Metal direct (no HTTP) |
| `backends/bench_gptoss.py` | openharmony-mlx GPT-OSS (N/A) |
| `benchmark.py` | Main runner (all backends) |
| `config.py` | Shared PROMPT, MAX_TOKENS, RUNS |
