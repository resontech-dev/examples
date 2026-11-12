# Qwen3.5-122B-A10B (GPTQ-Int4): level-2 text + vision assistant (vLLM)

The level-2 model on one card: a 122B mixture-of-experts with 10B active
parameters, scoring at the Claude Sonnet 4.6 level on GPQA Diamond and
HLE (levels: `reson_docs/docs/inference/models/README.md`). Text, images
and tool calls on one model id `assistant`, native 262k window,
Apache-2.0. It is the **switch option next to gpt-oss-120b**: the same
96 GB card or H200 serves either model, so a client who starts on
gpt-oss-120b moves up a level by changing `model_source`, not hardware.
What it costs: ~79 GB of Int4 weights instead of 65 GB MXFP4, so the KV
pool on a 96 GB card is smaller (about 7 GB against ~20 GB), and decode
reads 10B active parameters per token instead of 5B.

**Nothing in this folder has booted yet.** Every number below is an
estimate from the model card and the memory-bandwidth rule, written for
the day a 96 GB or H200 host is rented.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min -> rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 84 -> 96 GB | 16 | 128 GB | ~118 GB (image 15 + 1.3x79 GB checkpoint) | 96gb / 2x80gb |

Suggested GPUs: RTX PRO 6000 96 GB or H200 141 GB (TP=1, as configured);
2x H100 / A100 80 GB in one worker with TP=2 (three edits plus the
placement block, listed under "What to change"); official FP8 (~125 GB)
on 2x H200 TP=2.

Estimated capacity, KV about 25 KB/token (12 attention layers of 48):

| Host | KV pool | Cached tokens | Sessions at 30k | Decode per stream |
|---|---|---|---|---|
| RTX PRO 6000 96 GB | ~7 GB | ~280k | ~9 | 40-60 tok/s |
| H200 141 GB | ~49 GB | ~1.9M | ~60 | 100-150 tok/s |
| 2x H100 80 GB, TP=2 | ~60 GB pooled | ~2.4M | ~75 | 80-120 tok/s |

Measure with `llmbench.py` before quoting any of it.

## VERIFY before the first submit

1. **Checkpoint.** `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` is the official Int4
   repo name in the Qwen3.5 collection; confirm it exists. Fallback:
   `Qwen/Qwen3.5-122B-A10B-FP8` (~125 GB) on 2x H200.
2. **Int4 MoE kernel.** The expert layers run through vLLM's `moe_wna16`
   path, unverified on the pinned vLLM 0.18.0 (which ships the Qwen3.5
   model classes). Boot
   `../Qwen3.5-35B-A3B-GPTQ-Int4_agents_vision_min_22gb_vram` on a 24 GB
   card first: same kernel, same answer, no rental needed. If it fails
   with an unsupported-quantization error for FusedMoE, use the FP8
   checkpoint here.
3. **MTP speculative decoding.** The card lists `qwen3_next_mtp`; the block
   ships commented out. Boot without it, then add it and compare tok/s.
4. **Tool parser.** `qwen3_coder` is the official recipe; community reports
   `qwen3_xml` holds up better on long agentic runs.

## Client notes

Same client contract as the other Qwen3.5 jobs: one model id `assistant`
for text and images, thinking off per request
(`chat_template_kwargs: {enable_thinking: false}` in the body), the key as
both `Authorization: Bearer` and `X-API-Key`. On the 96 GB card there is
no room for `embed` / `rerank` sidecars: run them on another card (the
`Qwen3-8B-AWQ_agents_bundle` sidecars fit any 16 GB card). On an H200 the
sidecars fit at 0.04 each with the chat share at 0.84.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # text
python predict.py --image               # sample_data/invoice.png -> JSON
python predict.py --tools               # tool-calling round trip
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where | for what |
|---|---|
| `vllm.tensor_parallel_size: 2` + `gpus_per_worker: 2` (yaml) + `tensor_parallel_size=2` + `placement_group_config` (py) | 2x80 GB TP=2 shape |
| `model_source` | `Qwen/Qwen3.5-122B-A10B-FP8` on 2x H200 TP=2 if the Int4 kernel fails |
| `max_model_len`, `max_num_seqs` | 262144 / 32 on an H200; 65536 / 16 shipped for 96 GB |
| `speculative_config` (commented) | MTP decoding once verified |
| `limit_mm_per_prompt` | pages per request |
| `tool_call_parser` | `qwen3_coder` or `qwen3_xml` |

## License

Apache-2.0 (model and official checkpoints). No third-party re-upload is
involved as shipped.
