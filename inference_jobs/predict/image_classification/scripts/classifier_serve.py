"""Image classification template — image bytes -> top-K labels.

Request:  raw image bytes (jpeg / png / tiff) as POST body, or
          JSON {"url": "https://..."} / {"data_b64": "..."} (platform-resolved).

Response:
    {
      "predictions": [
        {"class_id": 285, "class_name": "Egyptian cat", "score": 0.842},
        ...
      ],
      "image_size": {"width": 1024, "height": 768},
      "device": "cuda",
      "elapsed_ms": 12.4
    }
"""
from __future__ import annotations

import io
import time
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


class ImageClassifier:
    def __init__(
        self,
        num_classes: int = 1000,
        image_size: int = 224,
        top_k: int = 5,
        class_names: list[str] | None = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.top_k = int(top_k)
        self.class_names = class_names
        self.device = _pick_device()
        self._weights_loaded_from: str | None = None

        # Build the architecture without downloading pretrained weights —
        # if the customer ships a .pt file, load_weights() replaces these.
        # If they don't, we fall back to ImageNet-pretrained ResNet50 so
        # the deploy still answers requests (you'll see ImageNet labels).
        self.model = resnet50(weights=None)
        if num_classes != 1000:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Fallback labels if customer didn't supply class_names AND didn't
        # ship weights. Only meaningful for num_classes == 1000.
        self._imagenet_labels: list[str] | None = None
        if class_names is None and num_classes == 1000:
            try:
                self._imagenet_labels = ResNet50_Weights.DEFAULT.meta["categories"]
            except Exception:
                self._imagenet_labels = None

    def load_weights(self, weights_path: str) -> None:
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        own = self.model.state_dict()
        filtered = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        self._weights_loaded_from = weights_path
        print(
            f"[ImageClassifier] loaded {len(filtered)}/{len(own)} tensors from {weights_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def _label_for(self, class_id: int) -> str:
        if self.class_names is not None and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        if self._imagenet_labels is not None and 0 <= class_id < len(self._imagenet_labels):
            return self._imagenet_labels[class_id]
        return f"class_{class_id}"

    @torch.inference_mode()
    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        orig_w, orig_h = img.size
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        scores, ids = probs.topk(min(self.top_k, probs.shape[0]))

        predictions = [
            {
                "class_id": int(i),
                "class_name": self._label_for(int(i)),
                "score": float(s),
            }
            for s, i in zip(scores.tolist(), ids.tolist())
        ]
        return {
            "predictions": predictions,
            "image_size": {"width": orig_w, "height": orig_h},
            "device": str(self.device),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def warmup(self) -> None:
        # One throwaway forward pass per replica so the first real
        # request doesn't pay the cuDNN autotune tax.
        x = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.inference_mode():
            self.model(x)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "device": str(self.device),
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "weights_source": self._weights_loaded_from or "untrained-default",
        }


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
