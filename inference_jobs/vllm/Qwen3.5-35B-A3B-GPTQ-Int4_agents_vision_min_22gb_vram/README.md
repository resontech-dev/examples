# Qwen3.5-35B-A3B (GPTQ-Int4): fast MoE text + vision assistant (vLLM)

The speed rung between the 9B and the 27B: a 35B mixture-of-experts with
3B active parameters per token. Same contract as the 27B job (one model
id `assistant` for text, images and tool calls, native 262k window,
Apache-2.0), decoding roughly 3x faster per card because only the active
parameters are read per token. Benchmarks put it a notch below the dense
27B on hard reasoning and about level on documents and drafting; the
share of answers accepted without edits is what decides, measure it on
your scenarios.

**Nothing in this folder has booted yet.** Every number below is an
estimate from the memory-bandwidth rule and the model card, written so the
job is ready the day a card is free.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min -> rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 22 -> 32 GB | 8 | 64 GB | ~41 GB (image 15 + 1.3x20 GB checkpoint) | 24gb / 32gb / 2x16gb |

Suggested GPUs: RTX 4090 / 3090 (24 GB, as configured, thin KV pool: about
16k context x 2 sequences); RTX 5090 / RTX PRO 4500 (32 GB, recommended,
~9 GB KV: 32k x 8 seqs); 2x RTX 5070 Ti 16 GB in one worker with TP=2
(our test bench, three edits listed under "What to change", ~10 GB of
weights per card).

Throughput, estimate only: on a 5090 roughly 3x the 27B's decode rate,
400-600 generated tok/s aggregate at 3k-token prompts, 15-20 users under
5 s to first token. Measure with `llmbench.py` before quoting.

## VERIFY before the first submit

1. **Checkpoint.** `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` is the official repo
   named in the vLLM recipe for this model; confirm it exists. Fallbacks
   are in `serve_module.py` (official FP8 at ~36 GB, community GPTQ).
2. **Int4 MoE kernel.** The expert layers run through vLLM's `moe_wna16`
   path, unverified on the pinned vLLM 0.18.0. The dense 27B and 9B jobs
   prove the Qwen3.5 hybrid architecture, not this kernel. A boot failure
   with an unsupported-quantization error for FusedMoE means: switch to the
   FP8 checkpoint. **This job is the cheap test for the same question on
   `../Qwen3.5-122B-A10B-GPTQ-Int4_agents_vision_min_84gb_vram`.** One boot
   here answers both.
3. **Tool parser.** `qwen3_coder` is the official recipe; community
   reports `qwen3_xml` holds up better on long agentic runs.

## Client notes

Same client contract as the 9B and 27B jobs: one model id `assistant` for
text and images, thinking off per request
(`chat_template_kwargs: {enable_thinking: false}` in the body), the key as
both `Authorization: Bearer` and `X-API-Key`. Sidecars (`embed`, `rerank`)
fit next to it only on the 32 GB card: copy the two `LLMConfig`s from
`../Qwen3-8B-AWQ_agents_bundle_min_12gb_vram` at 0.06 each and lower the
chat share to 0.78.

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
| `vllm.tensor_parallel_size: 2` + `gpus_per_worker: 2` (yaml) + `tensor_parallel_size=2` (py) | 2x16 GB shape (our 5070 Ti bench) or 2x24 GB |
| `model_source` | `Qwen/Qwen3.5-35B-A3B-FP8` on 2x24 GB TP=2 / 48 GB if the Int4 kernel fails |
| `max_model_len` | 16384 if a 24 GB card refuses to boot; up to 262144 on bigger cards |
| `limit_mm_per_prompt` | pages per request |
| `tool_call_parser` | `qwen3_coder` or `qwen3_xml` |
| `gpu_memory_utilization` | 0.92 shipped for 24 GB; 0.90 on 32 GB |

## License

Apache-2.0 (model and checkpoint). The community GPTQ fallback is a
third-party re-upload: vet the repo before selling an endpoint backed by it.
