# Qwen3.5-9B (AWQ) — text + vision assistant on one 16 GB card (vLLM)

Rung 2 of the small-over-big ladder
(`reson_docs/docs/inference/using-small-over-big-models/`, text and media):
**one model id for text tasks and image input**. Hybrid attention makes long dialogs and multi-page
requests cheap in KV; 262k native window; tool calling and optional thinking.
Apache-2.0.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 14 → 16 GB | 4 | 32 GB | ~31 GB (image 15 + 1.3×12 GiB checkpoint) | 16gb |

Suggested GPUs: RTX 5070 Ti / 4080 / 5080 (16 GB — as configured, chat
only). On 24 GB (4090 / 3090) add the embed + rerank sidecars from
`../Qwen3-8B-AWQ_agents_bundle_min_12gb_vram` with fractions
0.68 / 0.085 / 0.085, and raise `max_model_len` to 32k+.

## ⚠️ Checkpoint — verify before selling

No official 4-bit Qwen3.5-9B exists. Shipped: `QuantTrio/Qwen3.5-9B-AWQ`
(community, data-free AWQ, ~12 GiB, vLLM ≥ 0.16). Alternatives are listed in
`serve_module.py`; the safe fallback on 16 GB is the official bf16
`Qwen/Qwen3.5-4B` (~8 GB, weaker). The first run must be the
`how-to-benchmark.md` readiness checks on your own scenarios, not a demo.

## Client notes

- Images travel as standard OpenAI content parts (`predict.py --image`).
  Resize photos to 1 000–1 500 px on the long side client-side; ≤ 2 images
  per request as configured (`limit_mm_per_prompt`).
- The model thinks unless the request sends `chat_template_kwargs:
  {enable_thinking: false}` (`extra_body` in openai-python), as `predict.py`
  does. A server-wide default isn't possible on this stack
  (`reson_docs/docs/inference/vllm/known_issues.md`). Same auth as every job here: the key as both
  `Authorization: Bearer` and `X-API-Key`.
- One model id `assistant` for text and images: vary `temperature` per
  call (0 for extraction and classification, 0.2–0.3 for replies).

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # text
python predict.py --image               # sample_data/invoice.png → JSON
python predict.py --tools               # tool-calling round trip (qwen3_coder parser)
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `model_source` | the alternatives in the banner; `Qwen/Qwen3.5-4B` if AWQ misbehaves |
| `language_model_only=True` | text-only: frees ~1 GB, no image input |
| `tool_call_parser` | `qwen3_coder` (official) ↔ `qwen3_xml` (community, long agentic runs) |
| `limit_mm_per_prompt`, `max_model_len` | pages per request / context — the KV levers |
| `kv_cache_dtype="fp8"`, `enable_prefix_caching=True` | add one at a time on this hybrid model, verify output |

## Fine-tuning note

Qwen3.5 LoRA on the platform is bf16-only (no QLoRA in Unsloth yet) → a
24 GB card for training; the adapter then serves on this AWQ base via the
multi-LoRA block shown in the Qwen3-8B bundle. Status:
`reson_docs/docs/training/fine-tune-smaller-models/README.md`.
