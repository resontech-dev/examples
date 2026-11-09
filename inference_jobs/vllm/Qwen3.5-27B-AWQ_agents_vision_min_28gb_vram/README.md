# Qwen3.5-27B (AWQ) — premium text + vision assistant (vLLM)

Top rung of the small-over-big ladder
(`reson_docs/docs/inference/using-small-over-big-models/`, text and media):
harder drafting, long documents, fewer edits,
images on the same model id. Apache-2.0. Before buying this card, try the
9B with a LoRA adapter on the same scenarios — the share of answers
accepted without edits is often the same.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 28 → 32 GB | 8 | 64 GB | ~45 GB (image 15 + 1.3×22.5 GB checkpoint) | 32gb / 2x24gb |

Suggested GPUs: RTX 5090 / RTX PRO 4500 (32 GB — as configured);
2× RTX 4090 / 3090 in one worker with TP=2 (three edits, listed in
`inference.yaml` and `serve_module.py`). Official FP8 (~30 GB) needs
2×24 GB TP=2 or a 48 GB card; official bf16 (~54 GB) needs 2×32 or 80 GB.

Throughput, estimate from the memory-bandwidth rule (measure before
quoting): 5090 ≈ 0.6–0.8 req/s at 3k-token prompts, 160–220 generated
tok/s, 6–8 users under 5 s to first token — about 3× slower per card than
the 9B.

## ⚠️ Checkpoint — verify before selling

Shipped: `QuantTrio/Qwen3.5-27B-AWQ` (community, data-free AWQ, ~21 GiB).
No official 4-bit exists; alternatives are in `serve_module.py`. Run the
`how-to-benchmark.md` readiness checks on your own scenarios first.

## Client notes

Same client contract as the 9B job: one model id `assistant` for text and
images, thinking off per request, the key as both `Authorization: Bearer`
and `X-API-Key`. Sidecars (`embed`, `rerank`) fit on the same 32 GB card at
0.78 / 0.06 / 0.06 — copy their `LLMConfig`s from
`../Qwen3-8B-AWQ_agents_bundle_min_12gb_vram`.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # text
python predict.py --image               # sample_data/invoice.png → JSON
python predict.py --tools               # tool-calling round trip
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where | for what |
|---|---|
| `vllm.tensor_parallel_size: 2` + `gpus_per_worker: 2` (yaml) + `tensor_parallel_size=2` (py) | 2×24 GB shape |
| `model_source` | `Qwen/Qwen3.5-27B-FP8` on 2×24 GB TP=2 / 48 GB |
| `limit_mm_per_prompt`, `max_model_len` | pages per request / context |
| `tool_call_parser` | `qwen3_coder` ↔ `qwen3_xml` |

## Fine-tuning note

bf16 LoRA on 27B ≈ 56 GB: 2×32 GB or 1×80 GB — see
`reson_docs/docs/training/train_Qwen3.5-27B_plan.md`.
