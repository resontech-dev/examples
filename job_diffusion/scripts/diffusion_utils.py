"""Generic Stable Diffusion + LoRA + NVFlare adapter for image-generation FL jobs.

LoRA fine-tunes the UNet's attention layers — only ~10-50 MB flow per round
even on SDXL. Base UNet, VAE, and text encoder stay local on each worker.

Public API:
    load_pipeline(model_id)              -> StableDiffusionPipeline
    train(pipeline, data_root, ...)      -> LoRA state_dict (cpu)
    make_fl_adapter(model_id, ...)       -> fl_train_model

Shard layout:
    shard_X.zip
        images/<id>.jpg
        data.jsonl    # {"image": "images/<id>.jpg", "caption": "a photo of ..."}
"""
import json
import math
import os

import torch
import torch.nn.functional as F

# Compat shim
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{"weights_only": False, **kw})

from accelerate import Accelerator
from datasets import Dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer


def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Load the full SD pipeline; we'll only train the UNet."""
    return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)


def _apply_lora_to_unet(unet: UNet2DConditionModel, lora_rank: int = 16):
    """Wrap UNet with LoRA targeting attention projection layers."""
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    from peft import get_peft_model
    return get_peft_model(unet, config)


def _make_dataset(data_root: str, resolution: int):
    examples = []
    with open(os.path.join(data_root, "data.jsonl")) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def _preprocess(batch, tokenizer, image_root: str, resolution: int):
    tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    pixel_values = []
    for img_path in batch["image"]:
        img = Image.open(os.path.join(image_root, img_path)).convert("RGB")
        pixel_values.append(tf(img))
    inputs = tokenizer(
        batch["caption"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "pixel_values": torch.stack(pixel_values),
        "input_ids": inputs.input_ids,
    }


def train(model_id: str, data_root: str, out_dir: str, epochs: int, batch_size: int, lr: float, resolution: int = 512, lora_rank: int = 16, payload_weights=None) -> dict:
    """Train SD UNet LoRA on captioned images. Returns LoRA state_dict."""
    os.makedirs(out_dir, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze base, wrap UNet with LoRA
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet = _apply_lora_to_unet(unet, lora_rank=lora_rank)

    if payload_weights:
        tensor_weights = {k: torch.as_tensor(v) for k, v in payload_weights.items()}
        try:
            set_peft_model_state_dict(unet, tensor_weights)
        except Exception as e:
            print(f"set_peft_model_state_dict failed: {e}", flush=True)

    dataset = _make_dataset(data_root, resolution)
    tokenized = dataset.map(
        lambda b: _preprocess(b, tokenizer, data_root, resolution),
        batched=True, batch_size=8, num_proc=1,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type="torch", columns=["pixel_values", "input_ids"])

    loader = torch.utils.data.DataLoader(tokenized, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    for epoch in range(epochs):
        unet.train()
        for step, batch in enumerate(loader):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()
            pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(pred.float(), noise.float())
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            if step % 20 == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f}", flush=True)

    return {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(unet).items()}


def make_fl_adapter(model_id: str = "runwayml/stable-diffusion-v1-5", lora_rank: int = 16, resolution: int = 512, max_batch_size: int = 2, max_lr: float = 1e-4):
    """Build fl_train_model for a Stable Diffusion LoRA FL job."""

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        bs = min(int(env["batch_size"]), max_batch_size)
        lr = min(float(env.get("learning_rate", max_lr)), max_lr)

        weights = train(
            model_id=model_id,
            data_root=data_root,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=bs,
            lr=lr,
            resolution=resolution,
            lora_rank=lora_rank,
            payload_weights=payload.get("initial_weights"),
        )

        with open(os.path.join(data_root, "data.jsonl")) as f:
            samples = sum(1 for _ in f)
        return {"weights": weights, "samples": samples}

    return fl_train_model
