# vLLM jobs (`engine: vllm_openai`) — index & shared contract

Every folder here deploys an **OpenAI-compatible endpoint** via
`ray.serve.llm.build_openai_app`. Folder names end in `_min_<x>gb_vram` —
the minimum **total** VRAM the job's model(s) need (heaviest quantization,
short context). Per-job READMEs carry the full resource table.

Full knob reference: [../docs/openai_compatible_llm_flow.md](../docs/openai_compatible_llm_flow.md).

## Job index

| Folder | Model(s) | Shape | Min total VRAM | Tier |
|---|---|---|---|---|
| `Qwen2.5-Coder-7B_data_parallel_min_6gb_vram` | Qwen2.5-Coder-7B AWQ | DP×2 | 6 GB | 12gb ×2 |
| `Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit_pipeline_parallel_min_20gb_vram` | Qwen3-Coder-30B-A3B AWQ | PP=2 | 20 GB | 2x12gb |
| `Devstral-Small-24B_coding_min_16gb_vram` | Devstral Small 24B AWQ | single | 16 GB | 24gb |
| `Qwen3-Coder-480B-A35B_coding_template_min_260gb_vram` | Qwen3-Coder-480B-A35B AWQ | TP=4 **TEMPLATE** | 260 GB | frontier 4x80gb |
| `gpt-oss-20b_chat_min_14gb_vram` | gpt-oss-20b MXFP4 | single | 14 GB | 24gb |
| `gpt-oss-20b_rag_bundle_min_18gb_vram` | gpt-oss-20b + embed + rerank | 3 engines, 1 GPU | 18 GB | 24gb |
| `Mistral-Small-24B_chat_min_16gb_vram` | Mistral Small 24B AWQ | single | 16 GB | 24gb |
| `Mistral-Small-24B_rag_bundle_min_20gb_vram` | Mistral Small AWQ + embed + rerank | 3 engines, 1 GPU | 20 GB | 24gb |
| `GLM-4.5-Air_chat_tensor_parallel_min_62gb_vram` | GLM-4.5-Air 106B-A12B AWQ | TP=4 | 62 GB | 4x24gb |
| `GLM-4.5-Air_rag_bundle_tensor_parallel_min_66gb_vram` | GLM-4.5-Air AWQ + embed + rerank | TP=4 + 2 sidecars | 66 GB | 4x24gb |
| `Qwen2.5-7B_multi_lora_min_10gb_vram` | Qwen2.5-7B AWQ + N LoRA adapters | single, multi-LoRA | 10 GB | 12gb |
| `Mistral-Small-24B_multi_lora_min_16gb_vram` | Mistral Small AWQ + N LoRA adapters | single, multi-LoRA | 16 GB | 24gb |
| `Qwen2.5-VL-7B_vision_min_16gb_vram` | Qwen2.5-VL-7B fp16 | single, vision-in | 16 GB | 24gb |
| `Qwen2.5-VL-32B-AWQ_vision_tensor_parallel_min_36gb_vram` | Qwen2.5-VL-32B AWQ | TP=2 | 36 GB | 2x24gb |
| `whisper-large-v3_asr_min_6gb_vram` | Whisper large-v3 | single, ASR | 6 GB | 12gb |
| `Qwen3-8B-AWQ_guarded_rag_chat_min_22gb_vram` | Qwen3-8B AWQ + Qwen3Guard-8B + embed + rerank | 4 engines, 1 GPU | 22 GB | 24gb |
| `Qwen3-8B-AWQ_agents_bundle_min_12gb_vram` | Qwen3-8B AWQ + embed + rerank (+ Qwen3Guard-0.6B) | 3–4 engines, 1 GPU | 12 GB | 16gb |
| `Qwen3-8B-AWQ_media_studio_min_32gb_vram` | Qwen3-8B AWQ (plans, calls tools) + Z-Image-Turbo (`/v1/images/generations`) + Wan2.2-TI2V-5B (`/v1/videos`), Apache-2.0 **needs Ray 2.60, not booted** | vLLM engine + diffusers actor, 1 GPU (2 × 16 GB variant) | 32 GB | 32gb (RTX 5090) / 48gb |
| `Qwen3.5-9B-AWQ_assistant_vision_min_14gb_vram` | Qwen3.5-9B AWQ (text + vision, hybrid attention) | single | 14 GB | 16gb |
| `Qwen3.5-27B-AWQ_agents_vision_min_28gb_vram` | Qwen3.5-27B AWQ (text + vision) | single (TP=2 variant) | 28 GB | 32gb / 2x24gb |
| `whisper-large-v3-turbo_asr_min_4gb_vram` | Whisper large-v3-turbo | single, ASR | 4 GB | 12gb (shares) |
| `Qwen3.5-35B-A3B-GPTQ-Int4_agents_vision_min_22gb_vram` | Qwen3.5-35B-A3B GPTQ-Int4 (text + vision, 3B active) **written, not booted** | single (TP=2 variant) | 22 GB | 24gb / 32gb / 2x16gb |
| `Qwen3.5-122B-A10B-GPTQ-Int4_agents_vision_min_84gb_vram` | Qwen3.5-122B-A10B GPTQ-Int4 (text + vision) **written, not booted** | single on 96 GB (TP=2 variant) | 84 GB | 96gb / 2x80gb |
| `Qwen3.5-397B-A17B-FP8_multimodal_template_min_400gb_vram` | Qwen3.5-397B-A17B FP8 | TP=8 **TEMPLATE** | 400 GB | 8x80gb / 8xh200 |
| `gpt-oss-120b_chat_min_66gb_vram` | gpt-oss-120b MXFP4 **written, not booted** | single (TP=2 / TP=4 variants) | 66 GB | 80gb / 96gb / 4x24gb |
| `gpt-oss-120b_rag_bundle_min_70gb_vram` | gpt-oss-120b + embed + rerank **written, not booted** | 3 engines, 1 GPU | 70 GB | 96gb / h200 |
| `gpt-oss-safeguard-20b_policy_guard_min_14gb_vram` | gpt-oss-safeguard-20b MXFP4 (policy classifier) **written, not booted** | single | 14 GB | 16gb / 24gb |
| `Nemotron-3.5-Content-Safety_guard_vision_min_10gb_vram` | Nemotron 3.5 Content Safety 4B (text + image guard) **written, not booted** | single | 10 GB | 12gb / 16gb (shares) |
| `BrowseSafe_web_guard_min_34gb_vram` | BrowseSafe 30B-A3B fp8-on-load (HTML injection detector) **written, not booted** | single on 48 GB (TP=2 variant) | 34 GB | 48gb / 2x24gb / 80gb |
| `Qwen3-ASR-1.7B_asr_min_6gb_vram` | Qwen3-ASR-1.7B (EN / IT / PL / DE, no UK) **written, not booted** | single, ASR | 6 GB | 12gb (shares) |
| `Voxtral-Mini-4B-Realtime_asr_streaming_min_16gb_vram` | Voxtral Mini 4B Realtime, WebSocket `/v1/realtime` (13 languages, no UK), Apache-2.0 **written, not booted; realtime route through the router unproven** | single, streaming ASR | 16 GB | 16gb |
| `Qwen3-Embedding-4B_rag_sidecars_premium_min_11gb_vram` | Qwen3-Embedding-4B + Qwen3-Reranker-4B, fp8-on-load **written, not booted** | 2 pooling engines, 1 GPU | 11 GB | 12gb / 16gb |
| `Gemma-3-12B-W4A16_chat_vision_min_14gb_vram` | Gemma 3 12B W4A16 (Red Hat AI), text + vision, Gemma ToU **written, not booted** | single | 14 GB | 16gb |
| `Gemma-3-27B-W4A16_chat_vision_min_26gb_vram` | Gemma 3 27B W4A16 (Red Hat AI), text + vision, Gemma ToU **written, not booted** | single | 26 GB | 32gb |
| `Gemma-4-12B-QAT-W4A16_chat_vision_audio_min_14gb_vram` | Gemma 4 12B QAT W4A16 (Google), text + vision + audio, Apache-2.0 **needs Ray 2.60 (vLLM ≥ 0.23)** | single | 14 GB | 16gb |
| `Gemma-4-26B-A4B-QAT-AWQ_chat_vision_min_22gb_vram` | Gemma 4 26B-A4B MoE, community AWQ of Google QAT, Apache-2.0 **needs Ray 2.60 (vLLM ≥ 0.25)** | single | 22 GB | 24gb |
| `Gemma-4-31B-QAT-W4A16_chat_vision_min_28gb_vram` | Gemma 4 31B QAT W4A16 (Google), text + vision, Apache-2.0 **needs Ray 2.60 (vLLM ≥ 0.19.1)** | single | 28 GB | 32gb |
| `Qwen3.6-27B-FP8_agents_vision_min_40gb_vram` | Qwen3.6-27B official FP8 (text + vision), Apache-2.0 **needs Ray 2.60, not booted** | single (TP=2 variant) | 40 GB | 48gb / 96gb / 80gb |
| `Qwen3.6-35B-A3B-NVFP4_agents_vision_min_28gb_vram` | Qwen3.6-35B-A3B NVIDIA NVFP4 (text + vision, 3B active), Apache-2.0 **needs Ray 2.60 + Blackwell, not booted** | single | 28 GB | 32gb (RTX 5090) |
| `Qwen3.8-27B-FP8_agents_vision_min_40gb_vram` | Qwen3.8-27B official FP8 (text + vision), Apache-2.0 **needs Ray 2.60, not booted** | single (TP=2 variant) | 40 GB | 48gb / 80gb / 96gb / h200 |
| `Nemotron-3-Super-120B-A12B-NVFP4_agents_min_88gb_vram` | Nemotron 3 Super 120B-A12B NVFP4 (text, Mamba hybrid), NVIDIA Open Model License **needs Ray 2.60 + Blackwell, not booted** | single on 96 GB | 88 GB | 96gb (RTX PRO 6000) / B200 |
| `DeepSeek-V4-Flash_reasoning_template_min_180gb_vram` | DeepSeek V4-Flash 0731 (284B-A13B), MIT **TEMPLATE, needs Ray 2.60** | TP=4 + EP | 180 GB | 4xh200 / 4x80gb |
| `DeepSeek-V4-Pro_reasoning_template_min_900gb_vram` | DeepSeek V4-Pro 0813 (1.6T-A49B), MIT **TEMPLATE, needs Ray 2.60** | TP=8 + EP | 900 GB | 8xh200 / 8xb200 |
| `Kimi-K2.6_agents_vision_template_min_600gb_vram` | Kimi K2.6 INT4 QAT (1T-A32B, text + vision), Modified MIT, `trust_remote_code` **TEMPLATE, needs Ray 2.60** | TP=8 | 600 GB | 8xh200 / 8xb200 |
| `Mistral-Large-3-675B_chat_vision_template_min_690gb_vram` | Mistral Large 3 675B-A41B FP8 (text + vision, no thinking), Apache-2.0 **TEMPLATE, needs Ray 2.60** | TP=8 | 690 GB | 8xh200 / 8xb200 |
| `GLM-5.3_agents_template_min_760gb_vram` | GLM-5.3 FP8 (743B-A39B, text), glm-5.3 licence **TEMPLATE, needs Ray 2.60** | TP=8 | 760 GB | 8xh200 / 8xb200 |
| [`ukrainian_models/`](ukrainian_models/README.md) | MamayLM 12B/27B, Lapa 12B (Gemma 3, Ukrainian) | single int4 / TP=2 | 10–40 GB | own index |

Prod runs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Rows marked **needs Ray 2.60** use
features of that image and do not boot on older ones. Jobs without the marker were written against the
earlier Ray 2.55.1 / vLLM 0.18.0 image; version notes inside them (for example "vLLM 0.18 has no
`task`") describe that image.
Rows marked **written, not booted** were added 2026-11-12: the folders follow this contract and compile, but no
replica has started on hardware yet. Each README carries its VERIFY list; drop the marker here
after the first successful boot and benchmark.

## Small-over-big ladder

Which job to start from per task, smallest first. Every text job exposes one model id `assistant`, so a client
moves between rungs by changing the endpoint, not the code.

| Task | Rung 1 (16 GB card) | Rung 2 | Premium |
|---|---|---|---|
| Text + RAG | `Qwen3-8B-AWQ_agents_bundle_min_12gb_vram` | `Qwen3.5-9B-AWQ_assistant_vision_min_14gb_vram` · `gpt-oss-20b_chat_min_14gb_vram` (EN) · `Mistral-Small-24B_rag_bundle_min_20gb_vram` (IT, 24 GB) | `Qwen3.5-27B-AWQ_agents_vision_min_28gb_vram` |
| Documents / photos | `Qwen2.5-VL-7B_vision_min_16gb_vram` (AWQ, measured) | `Qwen3.5-9B-AWQ_assistant_vision_min_14gb_vram` (same model as text) | `Qwen2.5-VL-32B-AWQ_vision_tensor_parallel_min_36gb_vram` · `Qwen3.5-27B-AWQ_…` |
| Speech-to-text | `whisper-large-v3-turbo_asr_min_4gb_vram` · `../predict/Parakeet-TDT-0.6B-v3_asr_min_4gb_vram` (25 European languages incl. UK, NeMo; not booted) | `whisper-large-v3_asr_min_6gb_vram` · `Qwen3-ASR-1.7B_asr_min_6gb_vram` (EN / IT / PL / DE only, not booted) | + LoRA (`examples/training/nvflare/job_speech`) |
| Live speech-to-text | `../predict/Nemotron-3.5-ASR-Streaming-0.6B_asr_min_4gb_vram` (uk-UA among 19 locales; whole files until a live route exists; not booted) | `Voxtral-Mini-4B-Realtime_asr_streaming_min_16gb_vram` (no UK; WebSocket, not booted) | — |
| Who spoke when | `../predict/pyannote-community-1_diarization_min_2gb_vram` (offline diarization, pairs with any speech-to-text job; not booted) | — | — |
| Text-to-speech | `../predict/Piper-TTS_tts_cpu_min_0gb_vram` (uk / en / it, CPU) | `../predict/Orpheus-TTS-3B_tts_min_8gb_vram` (EN) | — |
| Images & video | `../predict/FLUX.2-klein-4B_imagegen_edit_min_16gb_vram` (images from text, edits from a screenshot or photo, Apache-2.0; not booted) · `../predict/Wan2.2-TI2V-5B_videogen_min_16gb_vram` (5 s clips from text or an image, no sound; not booted) | `../predict/LTX-2.3-22B_videogen_audio_min_16gb_vram` (video with sound, LTX-2 Community License; not booted) · `../predict/FLUX.1-schnell_imagegen_min_16gb_vram` (images; its 23.8 GB transformer needs a 24 GB+ card as written) | `Qwen3-8B-AWQ_media_studio_min_32gb_vram` (chat plans + images + video on one endpoint; not booted) · `../predict/Qwen-Image-2512-Edit-2511_imagegen_edit_template_min_80gb_vram` (klein's contract with 20B models: images from text + edits from 1–3 images, Apache-2.0, 80 GB card; not booted) |
| Ukrainian-tuned chat | `ukrainian_models/MamayLM-12B-W4A16_chat_min_10gb_vram` · `ukrainian_models/Lapa-12B-w4g128_chat_min_10gb_vram` | `ukrainian_models/MamayLM-12B-v2_chat_vision_tensor_parallel_min_26gb_vram` (2×16 GB) | `ukrainian_models/MamayLM-27B-v2-FP8_chat_vision_tensor_parallel_min_40gb_vram` (2×24 GB) |
| Fine-tuned adapters | `Qwen2.5-7B_multi_lora_min_10gb_vram` (shape; the agents bundle has the same block commented out) | `Mistral-Small-24B_multi_lora_min_16gb_vram` | — |
| Guardrails | `Qwen3-8B-AWQ_agents_bundle_min_12gb_vram` (`WITH_GUARD`, Qwen3Guard-0.6B) | `Qwen3-8B-AWQ_guarded_rag_chat_min_22gb_vram` (Qwen3Guard-8B) · `Nemotron-3.5-Content-Safety_guard_vision_min_10gb_vram` (images, written policy; not booted) | `gpt-oss-safeguard-20b_policy_guard_min_14gb_vram` (policy audit) · `BrowseSafe_web_guard_min_34gb_vram` (fetched pages); not booted |
| Premium RAG sidecars | `Qwen3-Embedding-4B_rag_sidecars_premium_min_11gb_vram` (fp8-on-load, not booted) | 8B siblings on 24 GB (README of the same job) | — |

On the Ray 2.60 image the Qwen3.6 jobs are the next rung up from their Qwen3.5 siblings: same
contract, higher scores (Qwen3.6-27B: GPQA 87.8, SWE-bench Verified 77.2 against 85.5 / 72.4). On a
96 GB card Qwen3.6-27B FP8 matches or beats Qwen3.5-122B-A10B on those benchmarks and leaves ~54 GB for
KV instead of ~6 GB.

Qwen3.8-27B is the rung above Qwen3.6-27B on the same card and contract (GPQA 89.2, SWE-bench Pro
61.7 against 87.8 / 53.5; Apache-2.0) and the strongest model here that fits one card. Its recipe
states vLLM ≥ 0.17, but a boot on the 0.29 image is still to be verified.

## How vLLM jobs differ from `../predict/` jobs

No `predict(data: bytes) -> dict` contract. `scripts/serve_module.py`
builds a Ray Serve app (`app = build_openai_app(...)`) and the deployed
cluster speaks the OpenAI protocol natively:

| Route | What |
|---|---|
| `POST /v1/chat/completions` | chat (streaming, tool calls, vision content parts) |
| `POST /v1/completions` | classic completions |
| `GET  /v1/models` | lists every `model_id` the cluster serves |
| `POST /v1/embeddings` | for `runner="pooling", convert="embed"` engines (RAG bundles) |
| `POST /v1/score` | for `runner="pooling", convert="classify"` engines (rerankers) |
| `POST /v1/audio/transcriptions` | for the Whisper job (validate on the pinned stack) |

Any OpenAI-compatible client (openai-python, curl, LangChain, Aider, Cline,
Continue) points `base_url` at `<endpoint>/v1` and works unmodified.

## The yaml ↔ Python sync rule

`inference.yaml`'s `vllm:` block and `serve_module.py`'s `engine_kwargs`
carry the **same two keys with the same names**:

```
vllm.tensor_parallel_size    == engine_kwargs.tensor_parallel_size
vllm.pipeline_parallel_size  == engine_kwargs.pipeline_parallel_size
```

yaml is what the scheduler reserves hardware from; Python is what vLLM
enforces at boot. The SDK refuses to submit on mismatch. Both are integers
≥ 1; TP must divide the model's attention-head count (stick to 1, 2, 4, 8).
TP spans GPUs **inside one worker** (PCIe/NVLink); PP spans **workers**
(one stage per worker, `workers_consumed = replicas × PP`).

## Requirements policy (why `requirements:` is empty everywhere)

vLLM jobs run the platform's prebuilt **ray-llm image**: `ray[serve,llm]==2.60.0` + `vllm==0.29.0`
+ torch 2.13/cu130 + transformers 5.17 (Python 3.12, driver R580+), baked as one tested set. The
image before it was Ray 2.55.1 + vLLM 0.18.0 + torch 2.10/cu128 + transformers 4.57.
Never re-declare ray / vllm / torch / transformers / numpy in a job's
`requirements:`; a customer `ray` line is rejected at submit. Extras only.

## Defaults every template ships (and why)

- **`kv_cache_dtype="fp8"`** — halves per-token KV cost ≈ doubles usable
  context/batch, negligible quality loss. Free 2×; on by default.
- **`enable_prefix_caching=True`** — chat/agent clients resend the whole
  history each turn; prefix caching makes turn N re-prefill only the new
  tail. This is *not* storage: KV blocks are ephemeral GPU memory, evicted
  under pressure, gone on restart, never written to disk — the stateless
  privacy story holds.
- **`enable_chunked_prefill=True`** — one user's giant prompt doesn't
  freeze everyone else's stream.
- **No `api_key` in `engine_kwargs`** — auth is edge-only (nginx serve-proxy
  validates `X-API-Key`); vLLM 0.18's `FrontendArgs.api_key` wants
  `list[str]` and a string crashes engine startup.
- **`accelerator_type=None`** — consumer GPUs carry no Ray accelerator
  label; any non-None value leaves replicas PENDING forever.
- **YaRN rope scaling** (24 GB+ tiers only): Qwen models trained at 32k
  stretch to ~131k via
  `hf_overrides={"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}`
  on Ray 2.55.1. transformers 5 (Ray 2.60) renamed the key: `{"rope_parameters": {…same…, "rope_theta": 1000000}}`.
  Quality degrades gently at the far end, and the window is only usable if
  the KV pool affords it (~74 KB/token fp8 on Qwen3-8B → 128k ≈ 9.5 GB for
  ONE conversation). Shipped commented-out where it applies.

## Tool calling (agentic chats & coding agents)

Set in `engine_kwargs` — the parser is **per model family**:

| Family | engine_kwargs | Status |
|---|---|---|
| Qwen2.5 / Qwen3 chat | `enable_auto_tool_choice=True, tool_call_parser="hermes"` | stable — on by default in our Qwen2.5 / Qwen3-8B jobs |
| Qwen3-Coder | `enable_auto_tool_choice=True, tool_call_parser="qwen3_coder"` | model card; with `hermes` the XML calls arrive as text. `qwen3_xml` if long inputs produce garbage |
| Mistral Small / Devstral | `enable_auto_tool_choice=True, tool_call_parser="mistral"` | stable — on by default in our Mistral jobs |
| Qwen3.5 / Qwen3.6 / Qwen3.8 (hybrid) | `enable_auto_tool_choice=True, tool_call_parser="qwen3_coder"` + `reasoning_parser="qwen3"` | official recipe. On vLLM 0.18 community reports `qwen3_xml` holds up better on long agentic runs; on 0.29 both names are one parser class |
| Llama family | `tool_call_parser="llama3_json"` | not used here (license) |
| gpt-oss | `enable_auto_tool_choice=True, tool_call_parser="openai"` (harmony → `tool_calls`) | in vLLM 0.18.0 and 0.29.0 (checked in source); on in the gpt-oss examples, off in the catalog presets |
| GLM-4.5 | `tool_call_parser="glm45"` | in vLLM 0.18.0 and 0.29.0; on in the GLM-4.5-Air examples |
| Gemma 4 | `tool_call_parser="gemma4"` + `reasoning_parser="gemma4"` + vendored `tool_chat_template_gemma4.jinja` | Ray 2.60 only |
| Nemotron 3 Super | `tool_call_parser="qwen3_coder"` + `reasoning_parser="nemotron_v3"` | Ray 2.60; no `super_v3` plugin needed |
| DeepSeek V4 | `tool_call_parser="deepseek_v4"` + `reasoning_parser="deepseek_v4"` + `tokenizer_mode="deepseek_v4"` | Ray 2.60 |
| Kimi K2.x | `tool_call_parser="kimi_k2"` + `reasoning_parser="kimi_k2"` | both images; K2.6 needs `trust_remote_code` |
| GLM-5.x | `tool_call_parser="glm47"` + `reasoning_parser="glm47"` | Ray 2.60 |
| Mistral Large 3 | `tool_call_parser="mistral"` + `tokenizer_mode` / `config_format` / `load_format="mistral"` | Ray 2.60; no reasoning parser |
| Guard / embed / rerank models | none — classifiers, not agents | — |

Authoritative check for the pinned image:

```bash
docker run --rm <registry>/ray-llm:2.60.0-py312-cu130 \
  vllm serve --help | grep -A3 tool-call-parser
```

On vLLM 0.29 a request that carries `tools` with `tool_choice` auto, required or named runs under a
strict tool-call grammar by default, and a `response_format` on the same request is dropped
(`VLLM_ENFORCE_STRICT_TOOL_CALLING=0` turns it off). `mistral` and `gemma4` have no such grammar.

A wrong parser name fails at engine boot, not at submit; run the check above for any parser not
in this table.

## Deploy (same for every job)

```bash
cd <job-folder>
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env        # RESON_API_KEY (rsk_… from /profile → Developer) + S3 keys
python submit.py            # uploads yaml + scripts/, prints endpoint + predict key
python predict.py           # job-specific smoke test
```

The SDK ships on PyPI (`resontech`, latest = what you want). The `>=0.2.1`
floor matters: 0.1.x is still on the index with an incompatible API
(`from_files`, email auth) — the floor upgrades stale environments instead
of failing on them. `requires-python >= 3.11`: on an older Python, pip says
"No matching distribution found", which reads like the package doesn't
exist — it does, switch interpreters.

Or paste `scripts/serve_module.py` + `inference.yaml` into the web wizard:
<https://beta.reson.tech/dashboard/inference/submit>.

## License flags

Everything in this index is Apache-2.0 or MIT **except** where a job README
says otherwise. Community AWQ quants are third-party re-uploads — vet the
repo (`VERIFY` comments in each serve_module) before selling an endpoint
backed by one.
