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
cd training/nvflare/job_diffusion
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

## Original training recipe (verbatim from `diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py`)

Source: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py

| Param | Value | Source |
|---|---|---|
| Optimizer | **AdamW** (β1=0.9, β2=0.999, ε=1e-8) | script default |
| Learning rate | **1e-4** | script default |
| LR schedule | **constant** | script default |
| Warmup steps | **500** | script default |
| Batch size | **16** per-device × grad_accum 1 (we clamp to 2 for FL memory) | script default |
| Epochs | num_train_epochs=100; max_train_steps overrides | script default |
| Resolution | **1024** (SDXL native) | script default |
| Weight decay | **1e-2** | script default |
| Grad clip | **max_grad_norm=1.0** | script default |
| Mixed precision | None default; usually `--mixed_precision="fp16"` or `"bf16"` | flag |
| LoRA | **rank=4** (default); UNet target_modules=["to_k","to_q","to_v","to_out.0"]; alpha=rank (PEFT default); dropout=0.0 | script default |
| Loss | MSE on noise prediction (epsilon-prediction; `--prediction_type=v_prediction` available) | script default |
| Optional Min-SNR weighting | `--snr_gamma` (None default; common: 5.0) | script flag |
| noise_offset | 0 | default |
| image augmentations | optional random crop + flip; resolution 1024 with lanczos resize | script default |
| validation | validation_epochs=1, num_validation_images=4 | script default |
| Hardware | typically 1× A100 40GB+ or 24GB consumer with bf16 + 8-bit Adam + grad_ckpt | inferred |

**FL config matches**: `learning_rate: 1e-4`, `batch_size: 1` (FL clamp; centralized recipe uses 16), `local_epochs: 20`, 5 rounds → effective ~100 epochs (matches centralized default). LoRA `r=4` set in `model_def.py`.

## Caveats

- **SDXL is the lightest VIABLE diffusion target for FL** — its predecessors (SD 1.5, SD 2.1) are easier hardware-wise but increasingly outdated; FLUX is too heavy.
- LoRA-only doesn't capture text-encoder updates. For new domain vocabulary, consider Textual Inversion alongside LoRA.
- `result.pt` is just the LoRA delta. To get a self-contained `.safetensors`, merge with the base UNet (see `pipe.fuse_lora()` in diffusers).
- The diffusers script defaults to **batch_size=16** but FL workers can rarely afford that for SDXL — we clamp to 1 per device. To keep effective batch comparable, raise `gradient_accumulation_steps` inside `diffusion_utils.py` to 16.
