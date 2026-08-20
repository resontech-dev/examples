"""SAM 2.1 promptable segmentation — image in + prompts, masks out.

Request (JSON via ``client.predict_json`` — application/octet-stream):

    {
      "image_b64": "<base64 png/jpeg>",      # or send raw image bytes instead
      "points": [[x, y], ...],               # optional prompt points (pixels)
      "labels": [1, ...],                    # 1=foreground, 0=background; pairs with points
      "boxes":  [[x1, y1, x2, y2], ...],     # optional box prompts
      "multimask": true                      # return top-K mask candidates (default)
    }

Raw image bytes (Content-Type image/*) are also accepted → segments
around the image center as a default prompt.

Response:

    {
      "masks": [
        {"score": 0.97, "area": 12345,
         "bbox": [x1, y1, x2, y2],
         "png_b64": "<base64 1-bit PNG, white = mask>"},
        ...
      ],
      "image_size": {"width": W, "height": H},
      "device": "cuda", "elapsed_ms": 42.0
    }

⚠️ VERIFY: uses the transformers Sam2 integration (>=4.56 —
Sam2Processor / Sam2Model, checkpoints facebook/sam2.1-hiera-*). Smoke
test on the pinned predictor image before selling; the official `sam2`
package is the fallback backend if the transformers path misbehaves.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


class Sam2Segmenter:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        max_masks: int = 3,
        dtype: str = "auto",
    ) -> None:
        from transformers import Sam2Model, Sam2Processor  # requires transformers>=4.56

        self.model_id = model_id
        self.max_masks = int(max_masks)
        self.device = _pick_device()
        torch_dtype = (
            torch.float16 if (dtype == "auto" and self.device.type == "cuda")
            else torch.float32
        )
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = (
            Sam2Model.from_pretrained(model_id, torch_dtype=torch_dtype)
            .to(self.device)
            .eval()
        )

    def warmup(self) -> None:
        img = Image.new("RGB", (256, 256), (128, 128, 128))
        with torch.inference_mode():
            self._segment(img, points=[[128, 128]], labels=[1], boxes=None, multimask=False)

    @torch.inference_mode()
    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        img, req = _parse_request(data)
        w, h = img.size

        points = req.get("points")
        labels = req.get("labels")
        boxes = req.get("boxes")
        multimask = bool(req.get("multimask", True))
        if not points and not boxes:
            points, labels = [[w // 2, h // 2]], [1]     # default: center point
        if points and not labels:
            labels = [1] * len(points)

        masks, scores = self._segment(img, points, labels, boxes, multimask)

        results = []
        for mask, score in sorted(zip(masks, scores), key=lambda t: -t[1])[: self.max_masks]:
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(mask)
            results.append({
                "score": float(score),
                "area": area,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "png_b64": _mask_to_png_b64(mask),
            })

        return {
            "masks": results,
            "image_size": {"width": w, "height": h},
            "device": str(self.device),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def _segment(self, img, points, labels, boxes, multimask):
        """Run one prompted forward pass → (masks [N,H,W] bool, scores [N])."""
        kwargs: dict[str, Any] = {}
        if points:
            # transformers prompt nesting: batch → object → points → xy
            kwargs["input_points"] = [[points]]
            kwargs["input_labels"] = [[labels]]
        if boxes:
            kwargs["input_boxes"] = [boxes]
        inputs = self.processor(images=img, return_tensors="pt", **kwargs).to(
            self.device, dtype=self.model.dtype
        )
        outputs = self.model(**inputs, multimask_output=multimask)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu().float(), inputs["original_sizes"].cpu()
        )[0]                                        # [num_prompts, K, H, W]
        scores = outputs.iou_scores.cpu().float()   # [batch, num_prompts, K]
        masks = masks.reshape(-1, *masks.shape[-2:]).numpy() > 0.5
        scores = scores.reshape(-1).tolist()
        return masks, scores

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_id": self.model_id, "device": str(self.device)}


def _parse_request(data: bytes):
    """JSON envelope with image_b64 + prompts, or raw image bytes."""
    try:
        req = json.loads(data.decode("utf-8"))
        if isinstance(req, dict) and "image_b64" in req:
            img_bytes = base64.b64decode(req["image_b64"])
            return Image.open(io.BytesIO(img_bytes)).convert("RGB"), req
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        pass
    return Image.open(io.BytesIO(data)).convert("RGB"), {}


def _mask_to_png_b64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
