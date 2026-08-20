"""FLUX.1-schnell — prompt in, PNG out.

Request (JSON via ``client.predict_json``):

    {"prompt": "a lighthouse at dawn, oil painting",
     "width": 1024, "height": 1024,      # optional, multiples of 16
     "steps": 4,                          # optional; schnell is 4-step distilled
     "seed": 42}                          # optional, reproducible when set

A bare text body is also accepted and treated as {"prompt": <body>}.

Response:

    {"image_b64": "<base64 PNG>", "width": 1024, "height": 1024,
     "steps": 4, "seed": 42, "elapsed_ms": 1900.0}

⚠️ FLUX.1-schnell is Apache-2.0; FLUX.1-dev is NON-COMMERCIAL — never
swap it in for a customer deploy (catalog license trap).
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

import torch


class FluxImageGen:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        offload: str = "auto",
        default_steps: int = 4,
        max_side: int = 1536,
    ) -> None:
        from diffusers import FluxPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX needs a CUDA GPU (device: gpu in inference.yaml)")

        self.model_id = model_id
        self.default_steps = int(default_steps)
        self.max_side = int(max_side)

        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        use_offload = offload == "on" or (offload == "auto" and total_vram_gb < 40)
        if use_offload:
            # Streams submodules through the GPU: fits ~16 GB VRAM at ~2×
            # latency. Needs big system RAM (the pipeline is ~24 GB).
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

        width = _snap(int(req.get("width", 1024)), self.max_side)
        height = _snap(int(req.get("height", 1024)), self.max_side)
        steps = min(int(req.get("steps", self.default_steps)), 12)
        seed = req.get("seed")
        generator = (
            torch.Generator("cuda").manual_seed(int(seed)) if seed is not None else None
        )

        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,          # schnell is distilled — CFG must stay 0
            max_sequence_length=256,
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
    """Clamp to [256, max_side] and snap to the model's 16-px grid."""
    return max(256, min(side, max_side)) // 16 * 16
