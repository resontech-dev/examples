# Qwen3 4B RAG sidecars: embeddings + reranker, no chat (vLLM)

The premium retrieval pair as **one deployment on one GPU**:
Qwen3-Embedding-4B (`/v1/embeddings`) and Qwen3-Reranker-4B (`/v1/score`),
nothing else. For clients whose chat model runs elsewhere (another card,
another job, a bigger node) or who want better recall than the 0.6B
sidecars inside the bundles. Both Apache-2.0, 32k windows, 100+ languages.

The model ids are the same `embed` / `rerank` every bundle uses, so an
Open WebUI or LiteLLM config moves from the 0.6B pair to this job by
changing the endpoint, not the settings.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Why 4B over 0.6B

Published numbers (Qwen3 Embedding report, MTEB; conditions in
`reson_docs/docs/benchmarks/inference/public_benchmarks_refference.md`).
Ours are measured only for the 0.6B pair so far.

| | 0.6B pair (in the bundles) | 4B pair (this job) |
|---|---|---|
| Embeddings, SciFact / NFCorpus nDCG@10 | 69.7 / 36.7 (ours: 70.2 / 36.3) | 78.3 / 41.1 |
| After reranking, SciFact / NFCorpus | 74.9 / 38.4 (ours: 76.4 / 38.9) | 78.4 / 41.6 |
| MTEB retrieval (embeddings) | 61.8 | 68.5 |
| Weights, bf16 | ~1.2 GB each | ~8 GB each |

The 8B siblings (Qwen3-Embedding-8B 69.4 MTEB, Qwen3-Reranker-8B) add a
point or two for twice the memory; a 24 GB card with fp8, same `model_source`
swap.

## Resources

| VRAM min->rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 11 -> 16 GB | 4 | 32 GB | ~37 GB (image 15 + 1.3x16 GB checkpoints) | 12gb / 16gb |

Suggested GPUs: RTX 5070 Ti / 4080 / 5080 (16 GB, as configured, fp8
weights); RTX 4090 / 3090 (24 GB) run the pair in bf16 (drop
`quantization`) or the 8B siblings in fp8. On a 12 GB card: shares 0.40,
`SIDECAR_KV_BYTES` 0.3 GiB, `max_model_len` 2048.

## The one rule that keeps this alive

**Fractions must sum to ~0.85, and each engine needs a fixed KV cache.** The
fraction reserves the engine's share of the card in Ray (`card_share`) and
checks free VRAM at boot; `kv_cache_memory_bytes` sets the cache, because
vLLM's automatic sizing counts other engines' VRAM as its own when they
start together. Shipped split: embed 0.42 / rerank 0.42, about 13.4 GB in
use on a 16 GB card.

## VERIFY before listing

- `quantization="fp8"` on pooling models (embed / classify runners) on
  vLLM 0.18.0 is unverified. If an engine refuses it at boot, drop the
  kwarg and run bf16 on a 24 GB card at the same shares.
- The optional third engine `rerank-mxbai` (mixedbread-ai/mxbai-rerank-large-v2,
  1.5B, Apache-2.0, BEIR avg 61.4 published) ships commented out in
  `serve_module.py` with the `Qwen2ForSequenceClassification` override from
  the vLLM scoring docs. A vLLM forum thread reports scores that differ
  from the mxbai library and a different order on large payloads: compare
  on BEIR (`reson_inference_benchmarks/quality`) before selling it.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv httpx   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py "How many vacation days do employees get?"
# -> embeds a toy corpus, cosine top-4, reranks, prints both orders with scores and timings
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Client notes

Embeddings: standard `client.embeddings.create(model="embed", input=[...])`,
up to 8k tokens per text as configured. Reranking: `POST /v1/score` with
`{"model": "rerank", "text_1": query, "text_2": [docs]}`; the openai
library has no score API, `predict.py` uses httpx. Send the key as both
`Authorization: Bearer` and `X-API-Key`. Pair with any chat job: the
bundles' `predict.py` shows the full retrieve-then-answer wiring.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `quantization="fp8"` (both engines) | drop for bf16 on a 24 GB card |
| `model_source` | `Qwen/Qwen3-Embedding-8B` / `Qwen/Qwen3-Reranker-8B` on a 24 GB card (fp8) |
| `card_share(...)` + matching `gpu_memory_utilization`, `SIDECAR_KV_BYTES` | rebalance; keep the sum <= 0.85 |
| `max_model_len` | 2048 on a 12 GB card; up to 32768 for long passages on a bigger card |
| `rerank_mxbai` block | third engine, see VERIFY above |

## License

Apache-2.0 for both models; mxbai-rerank-large-v2 is Apache-2.0 as well.
