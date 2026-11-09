# Agents bundle — Qwen3-8B + embed + rerank (+ guard) on one 16 GB card (vLLM)

The **minimum complete stack for an in-house assistant**: one chat model for
every text task (classification, Q&A, drafting, answer checks), embeddings +
reranker for RAG, and an optional 0.6B guard — three or four vLLM engines
behind one OpenAI endpoint on a single RTX 5070 Ti-class card. Apache-2.0
end to end. Rung 1 of the small-over-big ladder
(`reson_docs/docs/inference/using-small-over-big-models/`).

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 12 → 16 GB | 4 | 24 GB | ~26 GB (image 15 + 1.3×(5.9 + 2×1.2) GB) | 16gb |

Suggested GPUs: RTX 5070 Ti / 4080 / 5080 (16 GB — as configured). A 12 GB
card needs `CHAT_MAX_LEN = 8192`, `CHAT_MAX_SEQS = 4`, `CHAT_KV_GIB = 0.6` and
`SIDECAR_KV_GIB = 0.25`, which leaves almost no margin (not tested); a 24 GB
card gets ~32k context and room for the guard with `CHAT_KV_GIB` ≈ 8.

## Client notes

- Send the key as **both** `Authorization: Bearer` and `X-API-Key` (two auth
  layers — `predict.py` shows it). Strip `/predict` from the printed endpoint
  if present; the OpenAI surface is `<endpoint>/v1`.
- Qwen3 thinks unless the request turns it off, so send
  `chat_template_kwargs: {enable_thinking: false}` in the body (`extra_body`
  in openai-python). With thinking on, the answer still arrives in `content`
  and the thought in `reasoning`, but every call takes 2–5× longer and the
  thought counts towards `max_tokens`. A server-wide default isn't possible on
  this stack (`reson_docs/docs/inference/vllm/known_issues.md`).
- One model id `assistant` for every text task: vary `temperature` per call
  (0 for classification and checks, 0.2–0.3 for replies and drafts).

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv httpx   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                     # chat + embed + rerank (+ guard) in one pass
python predict.py --tools             # tool-calling round trip (hermes parser)
python predict.py --think "Поясни різницю між гарантією і сервісним обслуговуванням."
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Memory rule

Fractions sum to 0.85 of the card; the rest is CUDA contexts (~0.4 GB per
engine). KV cache sizes are fixed (`CHAT_KV_GIB`, `SIDECAR_KV_GIB`) because
vLLM's automatic sizing counts other engines' VRAM as its own when they start
together. Shipped for 16 GB:

| | assistant | embed | rerank | guard | chat context |
|---|---|---|---|---|---|
| `WITH_GUARD = False` (default) | 0.62 → 9.9 GB, KV ≈ 3.2 GB ≈ 43k tokens fp8 | 0.115 | 0.115 | — | 16k × 8 seqs |
| `WITH_GUARD = True` | 0.50 → 8.0 GB, KV ≈ 1.3 GB ≈ 17k tokens | 0.115 | 0.115 | 0.12 | 8k × 4 seqs |

Expected load on a 5070 Ti, from the Qwen2.5-Coder-7B-AWQ benchmark (same
size and quant, 3.2k-token prompts): ~1.4 req/s ceiling, plan at 1 req/s,
8 users under 5 s to first token, ~12.5M tokens/hour. Measure this bundle
the same way (`reson_inference_benchmarks`, `llmbench.py`) before quoting.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `WITH_GUARD` | add Qwen3Guard-Gen-0.6B as model `guard` |
| `FRACTIONS`, `CHAT_MAX_LEN`, `CHAT_MAX_SEQS`, `CHAT_KV_GIB`, `SIDECAR_KV_GIB` | rebalance — keep the sum ≤ 0.85 and the KV large enough for one `max_model_len` request |
| `kv_cache_dtype`, `enable_prefix_caching` | first two things to switch off if output is garbage (llmconfig.md → "Garbled output") |
| commented `enable_lora` + `lora_config` | step 2: serve fine-tuned adapters as `assistant:<name>` |
| `model_source` of `assistant` | `Qwen/Qwen2.5-7B-Instruct-AWQ`, `Qwen/Qwen3-4B-AWQ` — same shape, same parser |

## Next rungs

Text + vision in one model: `../Qwen3.5-9B-AWQ_assistant_vision_min_14gb_vram`.
Bigger: `../gpt-oss-20b_chat_min_14gb_vram` (EN reasoning),
`../Mistral-Small-24B_rag_bundle_min_20gb_vram` (IT, 24 GB),
`../Qwen3.5-27B-AWQ_agents_vision_min_28gb_vram` (premium, 32 GB).
