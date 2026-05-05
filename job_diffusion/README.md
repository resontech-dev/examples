# job_diffusion — Image Generation LoRA (SDXL + Pokémon)

Federated LoRA fine-tuning of **Stable Diffusion XL base** — the current de-facto SD standard with 2M+ monthly HF downloads. *(Important: SD 1.5 from runwayml is no longer available — Runway took down the repo in Aug 2024. SDXL is the replacement.)*

## Model

- **Name**: `stabilityai/stable-diffusion-xl-base-1.0`
- **HF link**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **Architecture**: U-Net (~3.5 B params) + dual text encoders (OpenCLIP-ViT/G + CLIP-ViT/L) + VAE
- **Library**: `diffusers` + `peft` + `accelerate`
- **LoRA targets**: `to_q`, `to_k`, `to_v`, `to_out.0` (U-Net attention only)

## Published metrics (verbatim from `stabilityai/stable-diffusion-xl-base-1.0` HF card)

The card's user-preference evaluation chart shows:
> SDXL 1.0 base > SDXL 0.9 > Stable Diffusion 2.1 > Stable Diffusion 1.5

(Exact percentage differences are visualized as a chart, not numerically tabulated. SD 2.1 and SD 1.5 cards likewise have no numeric FID/CLIP-score tables — only qualitative comparisons.)

## Why SDXL specifically

| Candidate | Status |
|---|---|
| **`stabilityai/stable-diffusion-xl-base-1.0`** ← chosen | 2M+ monthly downloads, 1.2k+ derived fine-tunes, official Stability AI release |
| `stabilityai/stable-diffusion-2-1` | Older, no numeric metrics on card |
| `runwayml/stable-diffusion-v1-5` | **Repository taken down** (Aug 2024, returns HTTP 401). Don't use as reference |
| `black-forest-labs/FLUX.1-schnell` | Newer (12 B), but heavy and license-restricted |

SDXL's **community adoption** is the strongest popularity signal — those 1.2k+ derived fine-tunes are the proof you can actually adapt SDXL with LoRA.

## Dataset

- **Name**: `lambdalabs/pokemon-blip-captions`
- **HF link**: https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
- **Format**: image (RGB) + BLIP-generated caption
- **Examples**: 833 captioned Pokémon illustrations
- **Domain**: pixel art / illustrated creatures

For different artistic styles, swap `HF_DATASET` in `build_shards_hf.py`:
- `m1guelpf/nouns` — Nouns DAO illustrations
- `recmek/anime-captions` — anime-style art
- Any HF image-caption dataset with `image` + `text`/`caption` fields

## Validation

- ✅ Static checks pass
- ⚠️ Simulator end-to-end: not run (diffusers not in test env)
- ⚠️ Platform: not yet run (heaviest job; needs 24 GB GPU minimum)

## Training cost (your FL scenario)

| Metric | Value |
|---|---|
| Per-step time (batch=2 × resolution=1024, SDXL UNet LoRA, fp16) | ~2-3 s on RTX 4090 (24 GB) |
| Per-round wall time | ~1.5-2 hours (200 images / 2 batch × 2 epochs, but SDXL is heavy) |
| **Total FL time** | **~8-12 hours** (5 rounds × 4 workers parallel) |
| Per-round transfer | ~20-30 MB (LoRA-only, rank 16, on UNet attention) |
| LoRA trainable params | ~6 M |
| Expected output | recognizable style transfer after 5 rounds (CFG 7.5 / 30 steps inference) |

## Hardware

- **Min GPU**: **24 GB VRAM** for SDXL at resolution 1024 with batch=2
- **Recommended**: A100 (40 GB) or A10G (24 GB) — anything below struggles
- **Per-worker disk**: ~10 GB (SDXL components ~7 GB + shard ~150 MB + activations + caches)

## Quick start

```bash
cd /home/pyaremenko/examples/job_diffusion
pip install datasets pillow diffusers transformers peft accelerate
python build_shards_hf.py   # downloads pokemon-blip-captions (~70 MB), builds 4 shards
```

If 24 GB is too much, drop `RESOLUTION = 768` in `model_def.py` (still SDXL native-supported, ~16 GB at batch=2).

## Output → SD inference

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Apply your federated LoRA delta to the UNet
pipe.unet.load_state_dict(torch.load("result.pt"), strict=False)

img = pipe(
    prompt="a fierce pokemon-style fire dragon, illustration",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
img.save("output.png")
```

## Why FL fits here

- **Multi-org artistic collaboration**: each studio has its own style; FL produces a global LoRA that captures all styles without exchanging proprietary illustrations
- **Privacy in personalized art generation**: per-user LoRAs that share weight updates without sharing source images
- **Distributed style libraries**: train shared style-LoRAs across many small datasets that individually wouldn't be enough to fine-tune SDXL

## Caveats

- **SDXL is the lightest VIABLE diffusion target for FL** — its predecessors (SD 1.5, SD 2.1) are easier hardware-wise but increasingly outdated; FLUX is too heavy.
- LoRA-only doesn't capture text-encoder updates. For new domain vocabulary, consider Textual Inversion alongside LoRA.
- `result.pt` is just the LoRA delta. To get a self-contained `.safetensors`, merge with the base UNet (see `pipe.fuse_lora()` in diffusers).
