"""Qwen-Image-2512 + Qwen-Image-Edit-2511: prompt in, image out; or image(s) + instruction in, edited image out.

Two 20B MMDiT models from the Qwen team, Apache-2.0, in one predictor.
Qwen-Image-2512 draws from text; Qwen-Image-Edit-2511 edits from 1–3
reference images. Both repos ship the same text encoder (Qwen2.5-VL-7B)
and VAE, byte for byte, so the job keeps one copy of them.

Request (JSON via ``client.predict_json``): the FLUX.2 klein job's fields
plus the two CFG knobs of a non-distilled model.

    {"prompt": "poster: 'GRAND OPENING Saturday 10:00', art-deco style",
     "images_b64": ["<PNG/JPEG>", ...],      # optional: 1–3 images → edit mode
     "width": 1328, "height": 1328,          # optional, snapped to 16 px
     "steps": 50,                             # optional; 50 for text-to-image, 40 for edits
     "true_cfg_scale": 4.0,                   # optional; 1 turns CFG off (2× faster, weaker)
     "negative_prompt": "...",                # optional; model-card defaults below
     "seed": 42,                              # optional; random when omitted, always returned
     "num_images": 1,                         # optional, at most max_images
     "format": "png"}                         # png | jpeg | webp

``image_b64`` (one image) is accepted as well as ``images_b64``. With an
image and no width/height, the output keeps the first image's aspect at
~1 MP. A bare text body is treated as {"prompt": <body>}.

Response:

    {"images_b64": ["<base64>", ...], "image_b64": "<the first one>",
     "mode": "t2i" | "edit", "width": 1328, "height": 1328, "steps": 50,
     "seeds": [42], "seed": 42, "format": "png", "model_id": "Qwen/Qwen-Image-2512",
     "elapsed_ms": 41000.0}
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import random
import threading
import time
from typing import Any

import torch

GRID = 16    # 8× VAE downsampling, then 2×2 latent patches
GIB = 1024**3

# Negative prompts from the model cards. Text-to-image: Qwen-Image-2512's
# list ("low resolution, low quality, deformed limbs, deformed fingers,
# oversaturated, waxy, faceless, over-smoothed, AI look, messy composition,
# blurred or distorted text"). Edits: a single space, as in the Edit-2511
# example (CFG on, nothing in particular to steer away from).
T2I_NEGATIVE = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
EDIT_NEGATIVE = " "


class QwenImageGenEdit:
    def __init__(
        self,
        t2i_model_id: str | None = "Qwen/Qwen-Image-2512",
        edit_model_id: str | None = "Qwen/Qwen-Image-Edit-2511",
        offload: str = "auto",
        t2i_steps: int = 50,
        edit_steps: int = 40,
        max_steps: int = 80,
        default_side: int = 1328,
        max_area: int = 1328 * 1328,
        edit_area: int = 1024 * 1024,
        max_refs: int = 3,
        max_images: int = 2,
    ) -> None:
        from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen-Image needs a CUDA GPU (device: gpu in inference.yaml)")
        if not t2i_model_id and not edit_model_id:
            raise ValueError("set t2i_model_id, edit_model_id or both")
        if offload not in ("auto", "on", "off"):
            raise ValueError(f"offload must be auto | on | off, got {offload!r}")

        self.t2i_model_id = t2i_model_id or None
        self.edit_model_id = edit_model_id or None
        self.t2i_steps = int(t2i_steps)
        self.edit_steps = int(edit_steps)
        self.max_steps = int(max_steps)
        self.default_side = int(default_side)
        self.max_area = int(max_area)
        self.edit_area = int(edit_area)
        self.max_refs = int(max_refs)
        self.max_images = int(max_images)
        self._gpu = threading.Lock()

        dtype = torch.bfloat16
        self.t2i = QwenImagePipeline.from_pretrained(self.t2i_model_id, torch_dtype=dtype) if self.t2i_model_id else None
        self.edit = None
        if self.edit_model_id:
            # Same text encoder and VAE files in both repos (equal sha256 on the
            # Hub): passed in, diffusers neither downloads nor loads them again.
            shared = {"text_encoder": self.t2i.text_encoder, "vae": self.t2i.vae} if self.t2i else {}
            self.edit = QwenImageEditPlusPipeline.from_pretrained(self.edit_model_id, torch_dtype=dtype, **shared)

        pipes = [p for p in (self.t2i, self.edit) if p is not None]
        self.weights_gib = round(_weights_gib(pipes), 1)
        vram_gib = torch.cuda.get_device_properties(0).total_memory / GIB
        # Resident needs every weight on the card plus ~12 GiB for activations
        # (1.7 MP with CFG): ~104 GiB for both models, ~66 GiB for one. Below
        # that, model CPU offload keeps one component on the GPU at a time; the
        # peak is one 38 GiB transformer plus activations.
        self.offloaded = offload == "on" or (offload == "auto" and vram_gib < self.weights_gib + 12)
        for p in pipes:
            p.set_progress_bar_config(disable=True)
            if not self.offloaded:
                p.to("cuda")
        # Offload hooks chain a pipeline's own components (text encoder →
        # transformer → VAE). The shared modules can carry one pipeline's hooks
        # at a time, so the pipeline about to run is re-hooked when it changes.
        self._hooked = None

    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        req = _parse_request(data)
        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("missing 'prompt' field")

        refs = _decode_images(req, self.max_refs)
        mode = "edit" if refs else "t2i"
        pipe = self.edit if refs else self.t2i
        if pipe is None:
            if refs:
                raise ValueError("this job has no edit model (edit_model_id is null): send a prompt without images")
            raise ValueError("this job has no text-to-image model (t2i_model_id is null): send an image to edit")

        width, height = self._size(req, refs)
        default_steps = self.edit_steps if refs else self.t2i_steps
        steps = max(1, min(int(req.get("steps") or default_steps), self.max_steps))
        cfg = float(req["true_cfg_scale"]) if req.get("true_cfg_scale") is not None else 4.0
        cfg = max(1.0, min(cfg, 10.0))
        negative = req.get("negative_prompt")
        if negative is None:
            negative = EDIT_NEGATIVE if refs else T2I_NEGATIVE
        negative = str(negative) if str(negative).strip() else " "
        count = max(1, min(int(req.get("num_images") or 1), self.max_images))
        base_seed = int(req["seed"]) if req.get("seed") is not None else random.randrange(2**31)
        fmt = str(req.get("format") or "png").lower()
        if fmt not in ("png", "jpeg", "webp"):
            raise ValueError(f"format must be png | jpeg | webp, got {fmt!r}")

        images_b64, seeds = [], []
        with self._gpu:
            if self.offloaded and self._hooked is not pipe:
                pipe.enable_model_cpu_offload()   # also moves all of its components to the CPU
                self._hooked = pipe
            try:
                with torch.inference_mode():
                    for i in range(count):
                        seed = base_seed + i
                        kwargs: dict[str, Any] = dict(
                            prompt=prompt,
                            negative_prompt=negative,
                            true_cfg_scale=cfg,
                            width=width,
                            height=height,
                            num_inference_steps=steps,
                            generator=torch.Generator("cpu").manual_seed(seed),
                        )
                        if refs:
                            kwargs["image"] = refs if len(refs) > 1 else refs[0]
                        image = pipe(**kwargs).images[0]
                        images_b64.append(_encode(image, fmt))
                        seeds.append(seed)
            except torch.cuda.OutOfMemoryError as exc:
                self._reset(pipe)
                raise RuntimeError(
                    f"out of GPU memory at {width}x{height} with {len(refs)} reference image(s); "
                    "lower the size, send fewer references, or set init_args.offload: \"on\""
                ) from exc
            except Exception:
                self._reset(pipe)
                raise

        return {
            "images_b64": images_b64,
            "image_b64": images_b64[0],
            "mode": mode,
            "width": width, "height": height, "steps": steps,
            "seeds": seeds, "seed": seeds[0], "format": fmt,
            "model_id": self.edit_model_id if refs else self.t2i_model_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "t2i_model_id": self.t2i_model_id,
            "edit_model_id": self.edit_model_id,
            "offloaded": self.offloaded,
            "weights_gib": self.weights_gib,
        }

    def _size(self, req: dict[str, Any], refs: list) -> tuple[int, int]:
        if refs and req.get("width") is None and req.get("height") is None:
            # Keep the first image's aspect ratio at edit_area (the pipeline's own
            # default would follow the last image instead).
            w, h = refs[0].size
            aspect = w / h
            return _snap_size(
                round((self.edit_area * aspect) ** 0.5), round((self.edit_area / aspect) ** 0.5), self.max_area
            )
        return _snap_size(
            int(req.get("width") or self.default_side),
            int(req.get("height") or self.default_side),
            self.max_area,
        )

    def _reset(self, pipe) -> None:
        """After a failed call: put every component back where it belongs."""
        if self.offloaded:
            pipe.enable_model_cpu_offload()
            self._hooked = pipe
        torch.cuda.empty_cache()


def _weights_gib(pipes: list) -> float:
    seen: set[int] = set()
    total = 0
    for pipe in pipes:
        for module in pipe.components.values():
            if isinstance(module, torch.nn.Module) and id(module) not in seen:
                seen.add(id(module))
                total += sum(t.numel() * t.element_size() for t in module.parameters())
                total += sum(t.numel() * t.element_size() for t in module.buffers())
    return total / GIB


def _parse_request(data: bytes) -> dict[str, Any]:
    try:
        req = json.loads(data.decode("utf-8"))
        if isinstance(req, dict):
            return req
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {"prompt": data.decode("utf-8", errors="replace")}


def _decode_images(req: dict[str, Any], max_refs: int) -> list:
    from PIL import Image

    raw = req.get("images_b64")
    if raw is None:
        raw = [req["image_b64"]] if req.get("image_b64") else []
    if isinstance(raw, str):
        raw = [raw]
    if len(raw) > max_refs:
        raise ValueError(f"at most {max_refs} reference images, got {len(raw)}")
    images = []
    for i, item in enumerate(raw):
        if isinstance(item, str) and item.startswith("data:"):
            item = item.split(",", 1)[-1]      # accept data URLs from browsers
        try:
            blob = base64.b64decode(item, validate=False)
            images.append(Image.open(io.BytesIO(blob)).convert("RGB"))
        except (binascii.Error, ValueError, OSError) as exc:
            raise ValueError(f"images_b64[{i}] is not a base64 PNG/JPEG/WebP image") from exc
    return images


def _snap_size(width: int, height: int, max_area: int) -> tuple[int, int]:
    """Floor to the 16-px grid first, then scale down if still over max_area."""
    width, height = max(256, width // GRID * GRID), max(256, height // GRID * GRID)
    if width * height > max_area:
        scale = (max_area / (width * height)) ** 0.5
        width = max(256, int(width * scale) // GRID * GRID)
        height = max(256, int(height * scale) // GRID * GRID)
    return width, height


def _encode(image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "png":
        image.save(buf, format="PNG")
    else:
        image.save(buf, format=fmt.upper(), quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")
