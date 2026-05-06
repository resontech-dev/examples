from diffusion_utils import make_fl_adapter

# stabilityai/stable-diffusion-xl-base-1.0 — current de-facto SD standard.
# (SD 1.5 / runwayml repo was taken down by Runway in Aug 2024 — DO NOT use.)
# Card-published reference: SDXL > SDXL 0.9 > SD 2.1 > SD 1.5 in user-preference
# evals; ~3.5B UNet params + dual text encoders (OpenCLIP-ViT/G + CLIP-ViT/L).
# 2M+ monthly downloads, 1.2k+ derived fine-tunes.
# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# Per diffusers `train_text_to_image_lora_sdxl.py` defaults:
#   AdamW (β1=0.9, β2=0.999, ε=1e-8), LR=1e-4, schedule=constant, warmup=500
#   batch=16 × grad_accum=1 (we clamp to 2 for FL memory budgets)
#   resolution=1024, max_grad_norm=1.0, weight_decay=1e-2
#   LoRA rank=4 (default), target_modules UNet=["to_k","to_q","to_v","to_out.0"]
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_RANK = 4         # diffusers script default
RESOLUTION = 1024     # SDXL native; needs 24+ GB VRAM. Drop to 768 if OOM.

fl_train_model = make_fl_adapter(BASE_MODEL, lora_rank=LORA_RANK, resolution=RESOLUTION)
