"""FLUX.2 [klein] 4B: prompt in, image out; or image(s) + instruction in, edited image out.

Request (JSON via ``client.predict_json``):

    {"prompt": "a lighthouse at dawn, oil painting",
     "width": 1024, "height": 1024,          # optional, snapped to 16 px
     "images_b64": ["<PNG/JPEG>", ...],      # optional: edit / reference images
     "steps": 4,                              # optional; the model is 4-step distilled
     "seed": 42,                              # optional; random when omitted, always returned
     "num_images": 1,                         # optional, at most max_images
     "format": "png"}                         # png | jpeg | webp

``image_b64`` (one image) is accepted as well as ``images_b64``. With an
image and no width/height, the output keeps the first image's aspect ratio.
A bare text body is treated as {"prompt": <body>}.

Response:

    {"images_b64": ["<base64>", ...], "image_b64": "<the first one>",
     "mode": "t2i" | "edit", "width": 1024, "height": 1024, "steps": 4,
     "seeds": [42], "seed": 42, "format": "png", "elapsed_ms": 2300.0}
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

GRID = 16


class KleinImageGen:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        offload: str = "auto",
        default_steps: int = 4,
        max_steps: int = 8,
        default_side: int = 1024,
        max_area: int = 1536 * 1536,
        max_ref_area: int = 1024 * 1024,
        max_refs: int = 4,
        max_images: int = 4,
    ) -> None:
        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 klein needs a CUDA GPU (device: gpu in inference.yaml)")

        self.model_id = model_id
        self.default_steps = int(default_steps)
        self.max_steps = int(max_steps)
        self.default_side = int(default_side)
        self.max_area = int(max_area)
        self.max_ref_area = int(max_ref_area)
        self.max_refs = int(max_refs)
        self.max_images = int(max_images)
        self._gpu = threading.Lock()

        self.pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        # Transformer (4B) and Qwen3 text encoder are ~8 GB each in bf16:
        # both resident need ~20 GB with activations. Below that, model CPU
        # offload keeps one of them on the GPU at a time.
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if offload not in ("auto", "on", "off"):
            raise ValueError(f"offload must be auto | on | off, got {offload!r}")
        self.offloaded = offload == "on" or (offload == "auto" and vram_gb < 22)
        if self.offloaded:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        req = _parse_request(data)
        prompt = str(req.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("missing 'prompt' field")

        originals = _decode_images(req, self.max_refs)
        refs = [_fit_area(img, self.max_ref_area) for img in originals]
        if refs and req.get("width") is None and req.get("height") is None:
            # Keep the first image's aspect ratio; small inputs come out at ~1 MP.
            ref_w, ref_h = originals[0].size
            if ref_w * ref_h > self.max_ref_area:
                ref_w, ref_h = refs[0].size
            area = min(self.max_area, max(ref_w * ref_h, self.default_side**2))
            aspect = ref_w / ref_h
            width, height = _snap_size(round((area * aspect) ** 0.5), round((area / aspect) ** 0.5), area)
        else:
            width, height = _snap_size(
                int(req.get("width") or self.default_side),
                int(req.get("height") or self.default_side),
                self.max_area,
            )
        steps = max(1, min(int(req.get("steps") or self.default_steps), self.max_steps))
        count = max(1, min(int(req.get("num_images") or 1), self.max_images))
        base_seed = int(req["seed"]) if req.get("seed") is not None else random.randrange(2**31)
        fmt = str(req.get("format") or "png").lower()
        if fmt not in ("png", "jpeg", "webp"):
            raise ValueError(f"format must be png | jpeg | webp, got {fmt!r}")

        images_b64, seeds = [], []
        with self._gpu, torch.inference_mode():
            for i in range(count):
                seed = base_seed + i
                kwargs: dict[str, Any] = dict(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=1.0,      # step-distilled: the pipeline ignores CFG
                    generator=torch.Generator("cpu").manual_seed(seed),
                )
                if refs:
                    kwargs["image"] = refs if len(refs) > 1 else refs[0]
                try:
                    image = self.pipe(**kwargs).images[0]
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"out of GPU memory at {width}x{height} with {len(refs)} reference image(s); "
                        "lower the size, send fewer references, or set init_args.offload: \"on\""
                    ) from exc
                images_b64.append(_encode(image, fmt))
                seeds.append(seed)

        return {
            "images_b64": images_b64,
            "image_b64": images_b64[0],
            "mode": "edit" if refs else "t2i",
            "width": width, "height": height, "steps": steps,
            "seeds": seeds, "seed": seeds[0], "format": fmt,
            "model_id": self.model_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_id": self.model_id, "offloaded": self.offloaded}


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


def _fit_area(image, max_area: int):
    """Downscale (never upscale) to max_area, dimensions on the 16-px grid."""
    from PIL import Image

    w, h = image.size
    scale = min(1.0, (max_area / (w * h)) ** 0.5)
    new_w = max(GRID, int(w * scale) // GRID * GRID)
    new_h = max(GRID, int(h * scale) // GRID * GRID)
    if (new_w, new_h) == (w, h):
        return image
    return image.resize((new_w, new_h), Image.LANCZOS)


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
