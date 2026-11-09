# Mistral Small 24B RAG bundle — chat + embed + rerank (vLLM)

Private RAG on **one GPU**: Mistral Small 24B (AWQ) + Qwen3-Embedding-0.6B
+ Qwen3-Reranker-0.6B behind a single OpenAI endpoint. The EU-friendly
variant of the RAG bundle — same wiring as the gpt-oss sibling, different
chat brain. All Apache-2.0.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 20 → 24 GB | 4 | 48 GB | ~36 GB (image 15 + 1.3×15.6 GB checkpoints) | 24gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured; chat
context capped at 8k because the sidecars eat the margin); RTX 5090
(32 GB) → raise chat to 0.70, `CHAT_KV_BYTES` to ~3 GiB and `max_model_len` to 32k.

## The one rule that keeps this alive

**Fractions must sum to ~0.85, and each engine needs a fixed KV cache.** The
fraction reserves the engine's share of the card in Ray (`card_share`) and
checks free VRAM at boot. `kv_cache_memory_bytes` sets the cache, because vLLM's
automatic sizing counts other engines' VRAM as its own when they start together.
Shipped split: chat 0.62 / embed 0.11 / rerank 0.11, about 21 GB in use.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py "Can I work remotely during probation?"
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Routes

Same three-route shape as the gpt-oss bundle: `assistant` on
`/v1/chat/completions`, `embed` on `/v1/embeddings`, `rerank` on
`/v1/score` (raw HTTP — see `predict.py`).

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| chat `model_source` | ⚠️ community AWQ — **vet before selling**; official bf16 needs 2×24 TP=2 |
| `card_share(...)` + the matching `gpu_memory_utilization`, `CHAT_KV_BYTES`, `SIDECAR_KV_BYTES` | rebalance — keep the sum ≤ 0.85 and the KV large enough for one `max_model_len` request |
| chat `max_model_len` | 8192 shipped (RAG turns are short); bigger card → bigger window |
| rerank `hf_overrides` | required as-is for Qwen3-Reranker |
