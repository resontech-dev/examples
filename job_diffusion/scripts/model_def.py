from diffusion_utils import make_fl_adapter

# stabilityai/stable-diffusion-xl-base-1.0 — current de-facto SD standard.
# (SD 1.5 / runwayml repo was taken down by Runway in Aug 2024 — DO NOT use.)
# Card-published reference: SDXL > SDXL 0.9 > SD 2.1 > SD 1.5 in user-preference
# evals; ~3.5B UNet params + dual text encoders (OpenCLIP-ViT/G + CLIP-ViT/L).
# 2M+ monthly downloads, 1.2k+ derived fine-tunes.
# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_RANK = 16
RESOLUTION = 1024     # SDXL native resolution; needs 24+ GB VRAM. Drop to 768 if OOM.

fl_train_model = make_fl_adapter(BASE_MODEL, lora_rank=LORA_RANK, resolution=RESOLUTION)
