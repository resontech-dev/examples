# gpt-oss-20b RAG bundle — chat + embed + rerank (vLLM)

The private-RAG stack ("ChatGPT for our documents") as **one deployment on
one GPU**: gpt-oss-20b chat + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B.
Ray Serve LLM's OpenAI router takes a *list* of LLMConfigs, so all three
engines share a worker — this is exactly the co-location the catalog sells
as a bundle (chat + embed + rerank, flat monthly). All Apache-2.0.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 18 → 24 GB | 4 | 48 GB | ~35 GB (image 15 + 1.3×15.4 GB checkpoints) | 24gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured);
RTX 5090 (32 GB) buys the chat engine a real KV pool — raise its
`gpu_memory_utilization` to 0.70 and `max_model_len` to 65536.

## The one rule that keeps this alive

**`gpu_memory_utilization` fractions must sum to ~0.85.** Each vLLM engine
pre-allocates its share at boot; three engines at the 0.9 default is an
instant OOM. Shipped split: chat 0.60 / embed 0.12 / rerank 0.12.

## Deploy & test

```bash
cp .env.example .env
python submit.py
python predict.py "How many vacation days do employees get?"
# → embeds a toy corpus, cosine top-5, reranks to top-3, answers with context
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Routes

| model_id | route | client call |
|---|---|---|
| `assistant` | `POST /v1/chat/completions` | `client.chat.completions.create(model="assistant", …)` |
| `embed` | `POST /v1/embeddings` | `client.embeddings.create(model="embed", input=[…])` |
| `rerank` | `POST /v1/score` | raw HTTP — see `predict.py` (`text_1`=query, `text_2`=docs) |

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| chat `model_source` | swap the chat model, keep the sidecars (see Mistral/GLM siblings) |
| the three `gpu_memory_utilization` values | rebalance chat context vs sidecars — keep the sum ≤ 0.85 |
| chat `max_model_len` | 16384 shipped (tight KV at 0.60); more needs a bigger card |
| rerank `hf_overrides` | required as-is for Qwen3-Reranker — vLLM's documented seq-cls conversion |
