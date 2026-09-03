# vLLM jobs (`engine: vllm_openai`) — index & shared contract

Every folder here deploys an **OpenAI-compatible endpoint** via
`ray.serve.llm.build_openai_app`. Folder names end in `_min_<x>gb_vram` —
the minimum **total** VRAM the job's model(s) need (heaviest quantization,
short context). Per-job READMEs carry the full resource table.

Full knob reference: `reson_docs/docs/inference/vllm/llmconfig.md`.
Model/tier catalog: `reson_docs/docs/inference/models-catalog.md`.
GPU fleet: `reson_docs/docs/inference/dc-hardware-requirements.md §2`.

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

## How vLLM jobs differ from `../predict/` jobs

No `predict(data: bytes) -> dict` contract. `scripts/serve_module.py`
builds a Ray Serve app (`app = build_openai_app(...)`) and the deployed
cluster speaks the OpenAI protocol natively:

| Route | What |
|---|---|
| `POST /v1/chat/completions` | chat (streaming, tool calls, vision content parts) |
| `POST /v1/completions` | classic completions |
| `GET  /v1/models` | lists every `model_id` the cluster serves |
| `POST /v1/embeddings` | for `task="embed"` engines (RAG bundles) |
| `POST /v1/score` | for `task="score"` engines (rerankers) |
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

vLLM jobs run the platform's prebuilt **ray-llm image** — `ray[serve,llm]==2.55.1`
+ `vllm==0.18.0` + torch 2.10/cu128 + transformers, baked as one tested set.
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
  `hf_overrides={"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}`.
  Quality degrades gently at the far end, and the window is only usable if
  the KV pool affords it (~74 KB/token fp8 on Qwen3-8B → 128k ≈ 9.5 GB for
  ONE conversation). Shipped commented-out where it applies.

## Tool calling (agentic chats & coding agents)

Set in `engine_kwargs` — the parser is **per model family**:

| Family | engine_kwargs | Status |
|---|---|---|
| Qwen chat / coder | `enable_auto_tool_choice=True, tool_call_parser="hermes"` | stable — on by default in our Qwen jobs |
| Mistral Small / Devstral | `enable_auto_tool_choice=True, tool_call_parser="mistral"` | stable — on by default in our Mistral jobs |
| Llama family | `tool_call_parser="llama3_json"` | not used here (license) |
| gpt-oss | harmony format — vLLM handles it natively | ⚠️ VERIFY whether the pinned image wants a parser flag at all |
| GLM-4.5 | `tool_call_parser="glm45"` | ⚠️ VERIFY — dedicated parser exists only in newer vLLM builds |
| Guard / embed / rerank models | none — classifiers, not agents | — |

Authoritative check for the pinned image:

```bash
docker run --rm <registry>/ray-llm:2.55.1-py311-cu128 \
  vllm serve --help | grep -A3 tool-call-parser
```

Rows marked ⚠️ ship **commented out** in `serve_module.py` — run the check,
then uncomment. A wrong parser name fails at engine boot, not at submit.

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
