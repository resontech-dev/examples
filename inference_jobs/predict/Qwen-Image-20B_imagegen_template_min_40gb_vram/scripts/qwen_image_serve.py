"""⚠️ TEMPLATE — Qwen-Image (20B MMDiT), prompt in, PNG out.

The scale-up image generator: best open text-rendering in images
(posters, UI mockups, signage with real typography). Frontier tier —
40–80 GB VRAM. Same request/response contract as the FLUX sibling so
clients swap endpoints, not code:

    {"prompt": "...", "width": 1328, "height": 1328,
     "steps": 50, "seed": 42, "negative_prompt": ""}
      → {"image_b64": "<PNG>", ...}

Before first real deploy: validate on the target card, tune
default_steps / true_cfg_scale from output quality, drop the banners.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

import torch


class QwenImageGen:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image",
        offload: str = "auto",
        default_steps: int = 50,
        max_side: int = 1664,
    ) -> None:
        from diffusers import DiffusionPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen-Image needs a CUDA GPU (frontier tier)")

        self.model_id = model_id
        self.default_steps = int(default_steps)
        self.max_side = int(max_side)

        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        use_offload = offload == "on" or (offload == "auto" and total_vram_gb < 60)
        if use_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.offloaded = use_offload

    @torch.inference_mode()
    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        req = _parse_request(data)
        prompt = (req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("missing 'prompt' field")

        width = _snap(int(req.get("width", 1328)), self.max_side)
        height = _snap(int(req.get("height", 1328)), self.max_side)
        steps = min(int(req.get("steps", self.default_steps)), 80)
        seed = req.get("seed")
        generator = (
            torch.Generator("cuda").manual_seed(int(seed)) if seed is not None else None
        )

        image = self.pipe(
            prompt=prompt,
            negative_prompt=req.get("negative_prompt", ""),
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=float(req.get("true_cfg_scale", 4.0)),
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {
            "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": width, "height": height, "steps": steps,
            "seed": seed, "model_id": self.model_id,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model_id": self.model_id,
            "offloaded": self.offloaded,
        }


def _parse_request(data: bytes) -> dict[str, Any]:
    try:
        req = json.loads(data.decode("utf-8"))
        if isinstance(req, dict):
            return req
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {"prompt": data.decode("utf-8", errors="replace")}


def _snap(side: int, max_side: int) -> int:
    return max(256, min(side, max_side)) // 16 * 16
