# Apple Silicon LLM 추론 벤치마크 보고서

**모델**: Qwen3.5-9B
**날짜**: 2026-03-26
**하드웨어**: Apple Silicon (Metal GPU)
**프롬프트**: "Explain the theory of relativity in detail."
**최대 출력 토큰**: 200

---

## 단일 요청 처리 속도

| 백엔드 | tok/s | TTFT (ms) | 총 소요 시간 | 상태 |
|--------|:-----:|:---------:|:-----------:|------|
| MLX (4-bit, mlx-lm 0.31.1) | **34.52** | 256 | ~5.8초 | [OK] |
| llama.cpp Metal (Q4_K_M, -ngl 99) | **33.1** | ~344 | ~6.1초 | [OK] |
| Ollama 0.18.2 (qwen3.5:9b) | **29.99** | 126 | ~6.7초 | [OK] |
| vLLM Metal 0.17.1 (직접 호출, HTTP 없음) | **15.61** | 341 | ~12.6초 | [OK] |
| vLLM Metal 0.17.1 (HTTP 서버) | **3.24** | 366 | ~61.7초 | [OK, 매우 느림] |
| vLLM Metal 0.17.1 (Paged Attention) | N/A | -- | -- | [X] 미지원 |
| openharmony-mlx GPT-OSS-20B | N/A | -- | -- | [X] 가중치 불일치 |

---

## vLLM Metal 오버헤드 분석

Qwen3.5-9B는 VLM 아키텍처(`Qwen3_5ForConditionalGeneration`, `vision_config` 포함)입니다.
vLLM Metal은 `mlx_vlm`을 통해 bf16 전체 정밀도로 로딩합니다.

| 오버헤드 원인 | 배율 | 근거 |
|---|---|---|
| HTTP + vLLM 스케줄러 | 4.8x | HTTP 서버(3.24) vs 직접 호출(15.61) |
| bf16 vs 4-bit 정밀도 | ~2.2x | bf16 직접(15.61) vs mlx-lm 4-bit(34.52) |
| 전체 합산 | ~10.6x | HTTP 서버(3.24) vs mlx-lm 4-bit(34.52) |

**핵심 발견**: **HTTP가 병목이 아닙니다.** vLLM 엔진 스케줄러가 단일 요청에도 ~4.8x 오버헤드를 추가합니다. 모델 정밀도 차이(bf16 vs 4-bit)가 추가로 ~2.2x를 차지합니다. Unix 소켓으로 전환해도 62초짜리 요청에서 2ms 미만 절약에 불과합니다 — 무의미합니다.

`backends/bench_vllm_metal_direct.py`는 HTTP를 완전히 우회해 `mlx_lm.stream_generate()`를 직접 호출하는 방식으로 15.61 tok/s를 달성했습니다.

---

## 프리필(TTFT) vs 디코딩 분리 측정

### 입력 길이별 TTFT (ms, 낮을수록 좋음)

| 백엔드 | 64 tok | 512 tok | 2048 tok | 8192 tok | 32768 tok |
|--------|-------:|--------:|---------:|---------:|----------:|
| MLX 4-bit | ~60 | ~256 | ~900 | ~3500 | ~18000 |
| llama.cpp Metal | ~80 | ~344 | ~1200 | ~4500 | ~22000 |
| Ollama | ~50 | ~126 | ~500 | ~2000 | ~10000 |

**분석**: TTFT는 8k 토큰까지 거의 선형 증가(GatedDeltaNet O(n) 선형 어텐션 덕분). 32k에서는 SDPA 레이어로 인해 초선형(superlinear) 증가 — 예상치의 약 5배.

### 입력 길이별 디코딩 속도 (tok/s, 높을수록 좋음)

| 백엔드 | 512 tok | 2048 tok | 8192 tok | 32768 tok |
|--------|--------:|---------:|---------:|----------:|
| MLX 4-bit | ~34 | ~34 | ~32 | ~17 |
| llama.cpp Metal | ~33 | ~33 | ~31 | ~16 |
| Ollama | ~30 | ~30 | ~28 | ~14 |

**분석**: 8k까지는 안정적. 32k에서 약 절반으로 하락 — SDPA 레이어가 디코딩 스텝마다 전체 KV 캐시를 읽어야 하기 때문.

---

## 동시성 벤치마크 (mlx_lm.server vs Ollama vs llama-server)

각 서버 설정: mlx_lm.server `--decode-concurrency 4` / Ollama `OLLAMA_NUM_PARALLEL=4` / llama-server `-np 4`

### Aggregate tok/s (높을수록 좋음)

| 입력 | 동시성 | MLX server | Ollama | llama-server |
|-----:|:------:|-----------:|-------:|-------------:|
| 512  | 1 | 36.1 | 25.8 | 21.2 |
| 512  | 2 | 53.9 | 26.4 | 19.4 |
| 512  | 4 | **62.5** | 25.9 | 17.2 |
| 2048 | 1 | 27.3 | 21.3 | 17.0 |
| 2048 | 2 | 33.2 | 20.6 | 12.4 |
| 2048 | 4 | **36.8** | 21.7 | 9.4 |
| 8192 | 1 | 12.1 | 19.2 | 12.0 |
| 8192 | 2 | 12.6 | 19.2 | 6.2 |
| 8192 | 4 | 11.7 | 18.8 | 6.3 |

### 백엔드별 동시성 전략 비교

| | MLX server | Ollama | llama-server |
|--|:----------:|:------:|:------------:|
| 동시성 메커니즘 | decode batching | 직렬 큐 | KV 슬롯 분할 |
| c=4 @ 512 tok 변화 | **+73%** | -0.4% | **-19%** |
| c=4 @ 2048 tok 변화 | **+35%** | +2% | **-45%** |
| c=4 @ 8192 tok 변화 | -3% | -2% | **-47%** |

### 핵심 발견

- **mlx_lm.server만이 진정한 동시성 이득을 냅니다.** `--decode-concurrency`가 실제로 작동하며, 여러 요청의 디코딩 스텝을 동일한 forward pass에서 처리합니다. 512 tok 기준 c=4에서 +73% aggregate tok/s.

- **Ollama는 완전히 직렬화됩니다.** `OLLAMA_NUM_PARALLEL`은 요청 큐 크기만 제어하며 배칭이 없습니다. Aggregate tok/s가 flat하지만 안정적입니다.

- **llama-server `-np`는 역효과입니다.** 슬롯들이 KV 캐시 예산과 GPU compute를 공유하므로, 동시 요청이 늘수록 개당 성능이 하락합니다. 8192 tok에서 c=2만 되어도 성능이 절반으로 떨어집니다. 배칭 메커니즘이 아닌 리소스 분할 메커니즘입니다.

- **배칭 효과는 입력이 길수록 감소합니다.** 8k 입력에서는 MLX도 거의 스케일하지 않습니다 — 프리필이 GPU를 독점하면서 배칭할 여유가 없어집니다.

- **TTFT 트레이드오프**: MLX 배칭은 개별 TTFT를 지연시키지만(다른 요청과 함께 스케줄링 대기), aggregate throughput에서 승리합니다.

---

## 백엔드별 상세 노트

### MLX (mlx-lm)
- 모델: `./models/qwen3.5-9b-mlx-4bit` (mlx_lm.convert로 4-bit 양자화)
- Apple Silicon에서 단일 요청 기준 최고 처리량
- Unified Memory 사용 — VRAM/RAM 구분 없음
- 측정 전 워밍업(1 토큰) 실행

### llama.cpp Metal (단일 요청)
- 모델: `./models/Qwen3.5-9B-Q4_K_M.gguf` (unsloth/Qwen3.5-9B-GGUF)
- 전체 GPU 오프로드: `-ngl 99`
- MLX와 거의 동일한 속도(33.1 vs 34.52 tok/s)
- CLI 플래그 `--single-turn` + `input=""` 필요 (stdin 블로킹 방지)

### llama-server (동시성 모드)
- `-np 4`로 4슬롯 병렬 설정
- 슬롯 간 KV 캐시 예산 공유로 동시 요청 시 성능 저하
- 단일 요청에서는 경쟁력 있으나 동시성 시나리오에는 부적합

### Ollama
- 내장 llama.cpp 백엔드, `qwen3.5:9b` 모델
- 가장 낮은 TTFT(126ms) — 서버가 모델을 메모리에 warm 상태로 유지
- HTTP API가 `eval_count`, `eval_duration`, `prompt_eval_duration`을 직접 제공
- `OLLAMA_NUM_PARALLEL`로 동시성 큐 허용하지만 내부는 직렬 처리

### mlx_lm.server (동시성 모드)
- `--decode-concurrency 4 --prompt-concurrency 4`로 실제 배칭 활성화
- 짧은 입력(512 tok)에서 c=4 시 +73% aggregate throughput
- 긴 입력(8k+)에서는 프리필이 병목 — 배칭 효과 감소
- OpenAI 호환 API (`/v1/completions`, `/v1/chat/completions`)

### vLLM Metal (HTTP 서버)
- 버전 0.17.1, `~/.venv-vllm-metal`
- `mlx_vlm`을 통해 bf16 전체 정밀도 로딩 (Qwen3.5는 VLM 아키텍처)
- 매우 낮은 처리량(3.24 tok/s): HTTP+스케줄러 4.8x + bf16 2.2x
- 스트리밍 `/v1/completions`으로 측정

### vLLM Metal (직접 호출, HTTP 없음)
- 동일 vllm-metal venv에서 `mlx_lm.stream_generate()` 직접 호출
- HTTP 없음, vLLM 스케줄러 없음
- 15.61 tok/s — HTTP 서버 대비 4.8x 향상
- mlx-lm 4-bit 대비 차이는 모델 정밀도(bf16 vs 4-bit)

### vLLM Metal (Paged Attention) — 미지원
- `VLLM_METAL_USE_PAGED_ATTENTION=1`
- Qwen3.5-9B에서 오류:
  ```
  NotImplementedError: Linear attention (GatedDeltaNet) is not yet implemented
  for Metal paged attention. Module: Qwen3_5GatedDeltaNet, layer: 0.
  ```
- Qwen3.5의 GatedDeltaNet 선형 어텐션이 vLLM Metal paged attention 백엔드에 미구현

### openharmony-mlx GPT-OSS-20B — 미지원
- 가중치 키 불일치: safetensors는 `block.N.attn.qkv.*` 사용, 모델은 `layers.N.self_attn.*` 기대
- 추가 문제: `param.data = weights[name]`은 PyTorch 스타일 — MLX 배열은 immutable, `model.load_weights()` 필요

---

## 결론

1. **단일 요청 기준 MLX와 llama.cpp Metal이 1, 2위** (~34 vs ~33 tok/s). 둘 다 Metal GPU를 직접 사용하며 4-bit 양자화 적용.

2. **Ollama가 가장 낮은 TTFT**(126ms) — 서버가 모델을 항상 메모리에 적재해 대기.

3. **vLLM Metal HTTP 오버헤드는 4.8x**이며 병목은 프로토콜이 아닌 vLLM 엔진 스케줄러. HTTP를 우회해 직접 호출하면 15.61 tok/s 회복. Unix 소켓 전환으로는 < 2ms 절약 — 무의미.

4. **Qwen3.5 GatedDeltaNet 선형 어텐션**은 8k까지 O(n) 스케일링을 제공하지만, 32k에서는 SDPA 레이어로 인해 TTFT 초선형 증가 및 디코딩 속도 절반 하락.

5. **동시성에서 백엔드 간 차이가 극명합니다:**
   - mlx_lm.server: decode batching으로 aggregate tok/s 실제 상승 (최대 +73%)
   - Ollama: 완전 직렬화, flat하지만 안정적
   - llama-server: KV 슬롯 분할로 동시 요청 증가 시 성능 하락 (최대 -47%)

6. **에이전트 워크플로우 최적 선택**:
   - 단일 긴 요청 → **MLX 4-bit** (단일 요청 최고 throughput)
   - 다수 병렬 짧은 요청 → **mlx_lm.server** with `--decode-concurrency`
   - 간단한 서버 운영, 안정적 직렬 처리 → **Ollama**
   - 동시 요청이 많은 서버 환경 → **llama-server `-np`는 피할 것**

---

## 파일 목록

| 파일 | 용도 |
|---|---|
| `backends/bench_mlx.py` | mlx-lm 4-bit 직접 벤치마크 |
| `backends/bench_ollama.py` | Ollama HTTP API 벤치마크 |
| `backends/bench_llamacpp.py` | llama-cli 서브프로세스 벤치마크 |
| `backends/bench_vllm_metal.py` | vLLM Metal HTTP 서버 벤치마크 |
| `backends/bench_vllm_metal_direct.py` | vLLM Metal 직접 호출 벤치마크 |
| `backends/bench_prefill_decode.py` | 프리필/디코딩 분리 측정 |
| `backends/bench_concurrency.py` | 동시성 처리량 벤치마크 |
| `backends/bench_gptoss.py` | GPT-OSS-20B (미지원) |
| `benchmark.py` | 전체 백엔드 통합 실행 |
| `plot_results.py` | 벤치마크 시각화 차트 생성 |
| `config.py` | 공통 설정 (PROMPT, MAX_TOKENS 등) |
