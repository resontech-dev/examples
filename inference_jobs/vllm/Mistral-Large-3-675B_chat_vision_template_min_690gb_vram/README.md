# ⚠️ TEMPLATE: Mistral Large 3 675B Instruct, TP=8 on one 8× H200 node (text + image)

**Needs the Ray 2.60.0 image (vLLM 0.29.0, CUDA 13, driver R580+). Not
deployable on the current fleet.** Nothing here has booted; every number is
napkin math.

Mistral's 675B mixture-of-experts, 41B active per token, 256k context,
text + image, **Apache-2.0**. An instruct model: no thinking mode, so it
answers fast and predictably. Its case is the licence and the vendor: a
European, Apache-licensed frontier-size model. On reasoning benchmarks the
thinking models (DeepSeek V4, Kimi K2.6, GLM-5.3) are ahead.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources (planning numbers, unvalidated)

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 640 GB (8× H100, NVFP4) → 1,128 GB (8× H200, FP8) | 32 | 1 TB | ~900 GB local NVMe (image + 1.3×681.5 GB) | 8xh200 / 8xb200 / 8x80gb (NVFP4) |

| Node | Build | KV after weights |
|---|---|---|
| 8× H200, NVLink | FP8, as shipped | ~37 GB per GPU, ~530k tokens |
| 8× B200 | FP8 | ~1M tokens |
| 8× H100 | NVFP4 `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` (403 GB, Marlin on Hopper) | ~17 GB per GPU, ~250k tokens; quality drops above 64k context |
| 4× B200, TP=4 | NVFP4 | ~800k tokens |

## Before first deploy

1. Pre-stage the checkpoint on NVMe with checksums.
2. `vllm serve --help | grep -A3 -E 'tokenizer-mode|config-format|load-format|tool-call-parser'`
   in the image: `mistral` must be accepted for all four.
3. Do not add the Eagle draft: broken with multimodal targets on vLLM ≥ 0.28
   (vLLM #57630, #43298).

## Client notes

Model id `mistral-large-3`. No thinking switch. Card sampling: temperature
below 0.1. Send the repo's `SYSTEM_PROMPT.txt` as the system message.
Up to 10 images per request as standard `image_url` parts.

## Deploy (once the node exists)

```bash
pip install -U "resontech>=0.2.1" openai python-dotenv   # Python 3.11+
cp .env.example .env
python submit.py
python predict.py
python predict.py --tools
```

## Variants

| Variant | Edits |
|---|---|
| NVFP4 on 8× H100 | `model_source` to the `-NVFP4` repo, `min_compute_capability: 9.0` stays, `min_vram_gb_per_worker: 600`, `gpu_models=["h100_80gb"]` |
| NVFP4 on 4× B200 | the same repo, `tensor_parallel_size=4` in both files, `gpus_per_worker: 4` |
| fp8 KV | `kv_cache_dtype="fp8"`: more sessions per node, check output first |
