# Qwen3.8-27B (FP8): text + vision assistant for agents (vLLM)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+).** Written
2026-11-19; nothing in this folder has booted.

The successor of `../Qwen3.6-27B-FP8_agents_vision_min_40gb_vram` on the
same dense 27B hybrid body, released August 2026. The strongest
open-weights model that fits one card: 20 on the Artificial Analysis
Intelligence Index v5 (gpt-oss-120b 4, Nemotron 3.5 Super VL 14). Text,
images and video on one model id, 262k native window (1M with YaRN),
Apache-2.0.

Model card numbers (vendor's, not replicated):

| Benchmark | Qwen3.8-27B | Qwen3.6-27B | Claude Opus 4.6 Max |
|---|---|---|---|
| GPQA Diamond | 89.2 | 87.8 | 91.3 |
| HLE | 30.8 | 24.0 | 40.0 |
| SWE-bench Pro | 61.7 | 53.5 | 53.4 |
| Terminal-Bench 2.1 | 73.0 | 63.4 | 78.2 |
| OSWorld-Verified | 84.3 | 63.9 | 72.7 |
| OmniDocBench 1.5 | 91.1 | 89.4 | 86.6 |

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 40 → 80 GB (96 GB or H200 for 262k context) | 8 | 64 GB | ~70 GB (image + 1.3 × ~31 GB checkpoint) | 48gb / 80gb / 96gb / h200 / 2x24gb |

KV figures are estimates carried over from the Qwen3.6-27B job (same layer
layout); replace them with the `GPU KV cache size` line from the first boot.

| Card | KV after weights (estimate) | Configuration |
|---|---|---|
| 1× 48 GB (L40S, RTX 6000 Ada, RTX PRO 5000) | ~10 GB | as shipped: 64k × 8 |
| 1× H100 80 GB | ~38 GB | `max_model_len=131072`, `max_num_seqs=16` |
| 1× 96 GB (RTX PRO 6000) | ~54 GB | `max_model_len=262144`, `max_num_seqs=16` |
| 1× H200 141 GB | ~95 GB | `max_model_len=262144`, `max_num_seqs=16` |
| 8× H100 or 8× H200 node | per card as above | one copy per card: `num_workers` or replicas, not TP |
| 2× 24 GB, TP=2 | ~6 GB per card | three edits below; block FP8 on Ada at TP=2 is unverified |

Native FP8 needs compute capability 8.9 or newer (Ada, Hopper, Blackwell),
which the yaml enforces. The GPU catalog has no RTX PRO 6000 or RTX 6000
Ada entry yet, so `submit.py` lists L40S, H100 and H200.

## VERIFY on the first boot

1. The engine starts on stock vLLM 0.29.0. The recipe states vLLM ≥ 0.17.0
   and transformers ≥ 5.8.0 but runs the `vllm/vllm-openai:qwen38` image.
2. Read one real answer: `python predict.py`. Qwen3.6-FP8 had a report of
   output collapsing into `!!!` (vLLM #55291). On RTX cards (`sm_120`)
   block FP8 goes through the DeepGEMM copy vendored in vLLM 0.29: if the
   answer is garbage, add `"VLLM_USE_DEEP_GEMM": "0"` to a `runtime_env`
   env var block and redeploy.
3. `python predict.py --image` and `python predict.py --tools`. The recipe
   verified text serving only, so the image path is ours to prove.
4. Thinking on and off: `python predict.py --think` returns a `reasoning`
   field, the default call does not.
5. Checkpoint size on disk (the card does not print it) and the KV line
   from the log; correct the table above.

## Client notes

Same contract as the Qwen3.5 / 3.6 jobs: one model id `assistant` for text
and images, the key as both `Authorization: Bearer` and `X-API-Key`.

- Thinking is on by default; off per request with
  `chat_template_kwargs: {"enable_thinking": false}`.
- `reasoning_effort`: `xhigh` (default), `medium`, `low`. The card warns
  that lower effort does not always make multi-turn agent runs faster.
- Earlier thinking stays in the history by default (new in 3.8; opt-in in
  3.6). It helps long agent loops and costs context;
  `{"preserve_thinking": false}` keeps only the latest turn's thinking.
- Card sampling: thinking `temperature=1.0, top_p=0.95, top_k=20`;
  non-thinking `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
  `predict.py` uses low temperatures for repeatable smoke tests.

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
| `max_model_len=131072`, `max_num_seqs=16` | one H100 |
| `max_model_len=262144`, `max_num_seqs=16` | one 96 GB card or H200 |
| `vllm.tensor_parallel_size: 2` + `gpus_per_worker: 2` (yaml) + `tensor_parallel_size=2` (py) | 2×24 GB |
| `model_source="nvidia/Qwen3.8-27B-NVFP4"` or `"Inferact/Qwen3.8-27B-NVFP4"`, plus `enforce_eager=True` on one RTX 5090 (the recipe: OOM in CUDA graph capture without it) | Blackwell cards only; VERIFY before selling |
| `speculative_config={"method": "mtp", "num_speculative_tokens": 3}` with `enable_prefix_caching=False` | faster decode for single users; the 3.5 / 3.6 hybrids garbled with both on (vLLM #53912), not re-checked on 3.8 |
| `default_chat_template_kwargs={"reasoning_effort": "medium"}` or `{"enable_thinking": False}` | server-wide thinking default |
| YaRN `rope_parameters` (factor 4.0) + `max_model_len=1000000` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | 1M context on an H200; only when prompts are that long, the card says static YaRN can hurt short ones |
