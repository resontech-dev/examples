# Nemotron 3 Super 120B-A12B (NVFP4): text reasoning + agents on one 96 GB card (vLLM)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+) and a
Blackwell card.** Prod runs Ray 2.55.1 / vLLM 0.18.0 until the switch
described in `reson_docs/docs/big_models_experiments/ray2.60.0-upgrade.md`.
Written 2026-11-18; nothing in this folder has booted.

NVIDIA's open 120B mixture-of-experts with 12B active per token, hybrid
Mamba-2 + MoE + attention, 1M context (256k default), text only. GPQA 79.2,
HLE 18.3, SWE-bench 60.5 (OpenHands). Its strength is long context per GB:
only 8 of 88 layers keep KV (~4 KB per token in fp8), so one card holds
262k × 8 sessions. Licence: NVIDIA Nemotron Open Model License (not
Apache; read it before selling).

On hard reasoning and code, `../Qwen3.6-27B-FP8_agents_vision_min_40gb_vram`
scores higher on the same card (GPQA 87.8, SWE-bench Verified 77.2) and
reads images. Pick Nemotron for very long documents and many long
sessions per card, and measure both on your scenarios.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 88 → 96 GB, Blackwell | 16 | 128 GB | ~130 GB (image + 1.3×80.3 GB checkpoint) | 96gb (RTX PRO 6000) / B200 |

| Hardware | Variant | Notes |
|---|---|---|
| 1× RTX PRO 6000 96 GB | NVFP4, as shipped | the vLLM recipe's verified shape; the GPU catalog has no entry for this card yet, so `submit.py` lists only B200 |
| 1× B200 | NVFP4 | room for 1M context (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) |
| 2× H100 80 GB, TP=2 | FP8 `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (128.4 GB, ships KV scales) | the card's stated minimum; ~9 GB KV per card |
| 1× H200 | FP8 at `gpu_memory_utilization=0.95` | ~10 GB left; unverified |
| ≤ 48 GB | none | |

## VERIFY on the first boot

1. Set aside 20+ minutes: an 80 GB pull plus a kernel compile. `MAX_JOBS=4`
   keeps the compile from eating all host RAM (vLLM #54237).
2. `python predict.py` and `python predict.py --think`: content without
   thinking, then a `reasoning` field with thinking.
3. `python predict.py --tools`: a tool call through the `qwen3_coder`
   parser.
4. A long-document request near 200k tokens: the reason to pick this model.

## Client notes

One model id `assistant`. Thinking is on by default; per request, through
`chat_template_kwargs`:

| Key | Default | Effect |
|---|---|---|
| `enable_thinking` | true | false: answer without the `reasoning` field |
| `low_effort` | false | shorter thinking |
| `force_nonempty_content` | false | true for coding agents: the answer always lands in `content` |
| `truncate_history_thinking` | true | drops earlier thinking from the history |

Card sampling: temperature 1.0, top_p 0.95.

## Deploy & test

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py
python predict.py --tools
```

## What to change

| where | for what |
|---|---|
| `model_source="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"`, `tensor_parallel_size=2` in both files, `gpus_per_worker: 2`, `min_compute_capability: 9.0`, `min_vram_gb_per_worker: 150`, `gpu_models=["h100_80gb", "h200"]` | FP8 on 2× H100 |
| `speculative_config={"method": "mtp", "num_speculative_tokens": 3}` with `enable_prefix_caching=False` | faster single-user decode; with prefix caching on, MTP garbles output on Mamba hybrids (vLLM #53912) |
| `max_model_len=1048576` + env `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | 1M context, B200 |
| `default_chat_template_kwargs={"enable_thinking": False}` | thinking off by default (works from Ray 2.58) |
