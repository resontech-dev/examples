# Qwen3-8B (AWQ) + embed — BFSI agent bundle, Stack B (vLLM)

The **chat half of the banking_and_insurance pilot**
([sectors/banking_and_insurance/bfsi_pilot](../../../sectors/banking_and_insurance/bfsi_pilot/README.md)):
the claims-assistant agent (`assistant`, hermes tool calling **on**) plus
the embedding engine (`embed`, Qwen3-Embedding-0.6B, 1024-dim → pgvector)
— two engines in one deployment on one 16 GB card.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 12 → 16 GB | 4 | 32 GB | ~24 GB (image 15 + 1.3×6.7 GB checkpoints) | 12gb / 16 GB card |

Suggested GPUs: RTX 5070 Ti / 4080 (16 GB — the pilot's target: chat gets
9.6 GB → 16k context × a few sessions); a 12 GB card fits at reduced
`max_model_len` (8192). Engine fractions: chat 0.60 + embed 0.12 = 0.72.

**One-GPU pilot rule:** Stack B and Stack A (the vision ingest job) run
**sequentially** — stop one before submitting the other. With a second
card both run permanently; nothing in the pilot's code changes.

## Deploy

```bash
cp .env.example .env        # RESON_API_KEY (rsk_…) + S3 keys
python submit.py            # prints CHAT_BASE_URL + CHAT_API_KEY for the pilot's .env
python predict.py           # chat smoke + embed dim=1024 check
python predict.py --tools   # hermes function-calling round trip
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## Routes

| model_id | route | pilot client |
|---|---|---|
| `assistant` | `POST /v1/chat/completions` (tools) | `agent_server.py` — the tool loop |
| `embed` | `POST /v1/embeddings` | `process_documents.py --embed`, `agent_server.py` search_docs |

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| chat `max_model_len` | 16384 shipped; 8192 on a 12 GB card |
| the two `gpu_memory_utilization` values | rebalance; keep sum ≤ 0.85 |
| `enable_auto_tool_choice` / `tool_call_parser` | **don't touch** — the agent loop dies without them |
