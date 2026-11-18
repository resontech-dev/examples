# Qwen3.6-27B (FP8): text + vision assistant for agents (vLLM)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+).** Prod
runs Ray 2.55.1 / vLLM 0.18.0 until the switch described in
`reson_docs/docs/big_models_experiments/ray2.60.0-upgrade.md`. Written
2026-11-18; nothing in this folder has booted.

The successor of `../Qwen3.5-27B-AWQ_agents_vision_min_28gb_vram` on the
same dense 27B body: GPQA 87.8, HLE 24.0, SWE-bench Verified 77.2 against
85.5 / 24.3 / 72.4 for Qwen3.5-27B. On those three benchmarks it matches or
beats Qwen3.5-122B-A10B (86.6 / 25.3 / 72.0) at a third of the memory, so
on a 96 GB card it leaves ~54 GB for KV instead of ~6 GB. Text and images
on one model id, 262k native window, Apache-2.0.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 40 → 48 GB (96 GB for 262k context) | 8 | 64 GB | ~70 GB (image + 1.3×30.9 GB checkpoint) | 48gb / 96gb / 2x24gb / 80gb |

| Card | KV after weights (estimate) | Configuration |
|---|---|---|
| 1× 48 GB (L40S, RTX 6000 Ada, RTX PRO 5000) | ~10 GB, ~160k tokens | as shipped: 64k × 8 |
| 1× 96 GB (RTX PRO 6000) | ~54 GB | `max_model_len=262144`, `max_num_seqs=16` |
| 1× H100 80 GB | ~38 GB | `max_model_len=131072`, `max_num_seqs=16` |
| 2× 24 GB, TP=2 | ~6 GB per card | three edits below; block FP8 on Ada at TP=2 is unverified |

Native FP8 needs compute capability 8.9 or newer (Ada, Hopper, Blackwell),
which the yaml enforces. The GPU catalog has no RTX PRO 6000 or RTX 6000
Ada entry yet, so `submit.py` lists L40S, H100 and H200; add the others
when the catalog does.

## VERIFY on the first boot

1. Read one real answer: `python predict.py`. vLLM #55291 reports FP8
   Qwen3.6 output collapsing into `!!!`. On RTX cards (`sm_120`) the
   block-FP8 matmuls go through the DeepGEMM copy vendored in vLLM 0.29,
   which we have not run on RTX yet: if the answer is garbage, first add
   `"VLLM_USE_DEEP_GEMM": "0"` to a `runtime_env` env var block and
   redeploy; if it persists, switch to the community int4 checkpoint
   (below) and report the image tag.
2. `python predict.py --image` and `python predict.py --tools`: the image
   path and a tool-call round trip.
3. Thinking on and off: `python predict.py --think` returns a `reasoning`
   field, the default call does not.

## Client notes

Same contract as the Qwen3.5 jobs: one model id `assistant` for text and
images, thinking on by default and off per request with
`chat_template_kwargs: {"enable_thinking": false}`, the key as both
`Authorization: Bearer` and `X-API-Key`. New in Qwen3.6:
`{"preserve_thinking": true}` keeps earlier thinking in the history, which
helps long agent loops. The model card advises a window of 128k or more
when thinking is on, because long reasoning needs room.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py                       # text
python predict.py --image               # sample_data/invoice.png → JSON
python predict.py --tools               # tool-calling round trip
```

## What to change

| where | for what |
|---|---|
| `max_model_len=262144`, `max_num_seqs=16` | one 96 GB card |
| `vllm.tensor_parallel_size: 2` + `gpus_per_worker: 2` (yaml) + `tensor_parallel_size=2` (py) | 2×24 GB |
| `model_source="Intel/Qwen3.6-27B-int4-AutoRound"` (19.0 GB) or `"cyankiwi/Qwen3.6-27B-AWQ-INT4"` (20.4 GB), `language_model_only=True` for text only | one 32 GB card (RTX 5090); community builds, VERIFY before selling |
| `model_source="nvidia/Qwen3.6-27B-NVFP4"` (21.9 GB, ships FP8 KV scales, so `kv_cache_dtype="fp8"` is safe there) | Blackwell cards only (RTX 5090, RTX PRO); NVIDIA's recipe for the 5090 runs 36k × 8 |
| `speculative_config={"method": "mtp", "num_speculative_tokens": 2}` with `enable_prefix_caching=False` | faster decode for single users; hybrid models garble with both on (vLLM #53912) |
| `default_chat_template_kwargs={"enable_thinking": False}` | thinking off by default (works from Ray 2.58) |
