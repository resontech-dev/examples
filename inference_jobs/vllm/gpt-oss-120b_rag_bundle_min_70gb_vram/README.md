# gpt-oss-120b RAG bundle: chat + embed + rerank (vLLM)

The private-RAG stack ("ChatGPT for our documents") on the premium chat
model, as **one deployment on one big card**: gpt-oss-120b chat +
Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B. Ray Serve LLM's OpenAI router
takes a *list* of LLMConfigs, so all three engines share a worker, the
co-location the catalog sells as a bundle. All Apache-2.0.

**Status: written, not booted.** Shares and KV sizes are estimates from the
checkpoint sizes; validate on the first 96 GB or H200 card.

Shared vLLM contract & defaults: [../README.md](../README.md).
Chat engine alone: [`../gpt-oss-120b_chat_min_66gb_vram`](../gpt-oss-120b_chat_min_66gb_vram).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 70 → 96 GB | 16 | 128 GB | ~103 GB (image 15 + 1.3×67.4 GB checkpoints) | 96gb / h200 |

| Card | chat share → KV | sidecars | Edits |
|---|---|---|---|
| RTX PRO 6000 96 GB | 0.78 → 6 GiB ≈ 340k tokens fp8 | 0.04 + 0.04 | as shipped (64k context, 16 seqs) |
| H200 141 GB | 0.60 → 16 GiB ≈ 900k tokens | 0.03 + 0.03 | `gpu_memory_utilization=0.60`, `CHAT_KV_BYTES = 16 GiB`, `max_model_len=131072`, `max_num_seqs=32`, sidecar shares 0.03 |
| H100 / A100 80 GB | does not fit | | 80×0.78 = 62 GB is below the 65 GB of weights; run the chat job alone here and the sidecars on another card |

## The one rule that keeps this alive

**Fractions must sum to ~0.86, and each engine needs a fixed KV cache.** The
fraction reserves the engine's share of the card in Ray (`card_share`) and
checks free VRAM at boot. `kv_cache_memory_bytes` sets the cache, because vLLM's
automatic sizing counts other engines' VRAM as its own when they start together.
Shipped split: chat 0.78 / embed 0.04 / rerank 0.04, about 82 GB in use on 96 GB.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv httpx   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py "How many vacation days do employees get?"
# → embeds a toy corpus, cosine top-5, reranks to top-3, answers with context
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

First boot downloads ~67 GB; pre-stage the chat checkpoint on the host when
you can.

## Routes

| model_id | route | client call |
|---|---|---|
| `assistant` | `POST /v1/chat/completions` | `client.chat.completions.create(model="assistant", …)` |
| `embed` | `POST /v1/embeddings` | `client.embeddings.create(model="embed", input=[…])` |
| `rerank` | `POST /v1/score` | raw HTTP — see `predict.py` (`text_1`=query, `text_2`=docs) |

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `card_share(...)` + the matching `gpu_memory_utilization`, `CHAT_KV_BYTES`, `SIDECAR_KV_BYTES` | rebalance chat context vs sidecars; keep the sum ≤ 0.86 and the KV large enough for one `max_model_len` request |
| chat `max_model_len` / `max_num_seqs` | 65536 / 16 shipped for 96 GB; 131072 / 32 on H200 |
| embed / rerank `model_source` | the 4B siblings (`Qwen3-Embedding-4B`, `Qwen3-Reranker-4B`) fit on H200 at share 0.08 each with `quantization="fp8"` |
| rerank `hf_overrides` | required as-is for Qwen3-Reranker — vLLM's documented seq-cls conversion |
| chat tool parser | commented VERIFY block, same as the chat job |

## Notes

- Guard: for user-facing endpoints add Qwen3Guard-Gen-0.6B (the agents
  bundle's `guard` LLMConfig, share 0.02 on this card) inline, and run
  [`../gpt-oss-safeguard-20b_policy_guard_min_14gb_vram`](../gpt-oss-safeguard-20b_policy_guard_min_14gb_vram)
  on another card for policy audits.
- Prefix caching on the chat engine: the RAG system prompt and the chat
  history are shared prefixes; ephemeral GPU memory, never on disk.
