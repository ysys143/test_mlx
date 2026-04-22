# Benchmark Report: Qwen3.5-9B + GPT-OSS

**Date**: 2026-04-22 15:49
**Prompt**: Explain the theory of relativity in detail, covering both special and general re...
**Max tokens**: 200
**Runs averaged**: 3

## Results

| Backend | tokens/sec        | TTFT (ms)         | total_sec          |
|---------|-------------------|-------------------|--------------------|
| omlx    | 19.02             | 7                 | 10.34              |

## Notes

- MLX: 4-bit quantized, Unified Memory, Apple Silicon optimized
- vLLM Metal (MLX path): MLX-managed KV cache, no paged attention
- vLLM Metal (Paged): VLLM_METAL_USE_PAGED_ATTENTION=1 (experimental)
- Ollama: llama.cpp-based server with HTTP overhead
- Ollama (MLX): Ollama >= 0.19, MLX engine on Apple Silicon (NVFP4), same HTTP API
- llama.cpp Metal: direct Metal GPU, Q4_K_M GGUF, ngl=99
- GPT-OSS-20B: MoE model (~3.6B active params/token), openharmony-mlx
- omlx: OpenAI-compat HTTP (localhost:8000), continuous batching + SSD KV cache, Apple Silicon
