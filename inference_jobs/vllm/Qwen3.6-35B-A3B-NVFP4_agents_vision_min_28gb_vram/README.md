# Qwen3.6-35B-A3B (NVFP4): fast MoE text + vision assistant (vLLM)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+) and a
Blackwell card.** Prod runs Ray 2.55.1 / vLLM 0.18.0 until the switch
described in `reson_docs/docs/big_models_experiments/ray2.60.0-upgrade.md`.
Written 2026-11-18; nothing in this folder has booted.

The successor of `../Qwen3.5-35B-A3B-GPTQ-Int4_agents_vision_min_22gb_vram`:
35B mixture-of-experts, 3B active per token, GPQA 86.0, HLE 21.4,
SWE-bench Verified 73.4 against 84.2 / 19.7 / 69.2 for Qwen3.5-35B-A3B.
Same contract: one model id `assistant` for text, images and tool calls,
262k native window, Apache-2.0. About 3× the decode speed of the dense
27B per card, a notch below it on hard reasoning.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 28 → 32 GB, Blackwell | 8 | 64 GB | ~50 GB (image + 1.3×23.4 GB checkpoint) | 32gb (RTX 5090) / 96gb / B200 |

Shipped as NVIDIA's NVFP4 build with the vLLM recipe flags for one RTX
5090: 64k window, fp8 KV (the checkpoint ships KV scales), Marlin MoE,
FlashInfer attention. NVFP4 is a Blackwell format; the yaml asks for
compute capability 10.0 or newer.

| Card | Variant | KV after weights (estimate) |
|---|---|---|
| RTX 5090 32 GB | NVFP4, as shipped | ~2.3 GB at 0.85: 64k window, several 16k chats |
| 48 GB (L40S, RTX 6000 Ada), H100 | official FP8 `Qwen/Qwen3.6-35B-A3B-FP8` (37.5 GB) | ~4 GB on 48 GB, ~30 GB on H100 |
| 24 GB (RTX 4090) | `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` (21.5 GB), `language_model_only=True`, 0.95, no MTP | ~1.5–2.5 GB; Intel warns of endless-loop reports. Boot and read answers before offering it |
| 16 GB | none on stock vLLM 0.29.0 | the 2-bit builds need a patched vLLM or leave no KV |

## VERIFY on the first boot

1. `python predict.py`: a coherent Ukrainian answer, not garbage.
2. `python predict.py --image` and `python predict.py --tools`.
3. The boot log lists `quantization=modelopt_fp4`, `kv_cache_dtype=fp8`
   and the FlashInfer backend. If an engine argument is rejected, check
   the name in `vllm serve --help` inside the image and report it.

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
| `model_source="Qwen/Qwen3.6-35B-A3B-FP8"`; drop `quantization`, `kv_cache_dtype`, `block_size`, `moe_backend`, `attention_backend`, `attention_config` and the env var; `gpu_memory_utilization=0.92`; yaml `min_compute_capability: 8.9`, `min_vram_gb_per_worker: 44` | FP8 on a 48 GB card or H100 |
| `language_model_only=True` | text only, frees the vision tower |
| `default_chat_template_kwargs={"enable_thinking": False}` | thinking off by default (works from Ray 2.58) |
| MTP | stays off: garbles with prefix caching on hybrid models (vLLM #53912) and fails to load on some 4-bit MoE builds (#63598, fixed after 0.29.0) |
