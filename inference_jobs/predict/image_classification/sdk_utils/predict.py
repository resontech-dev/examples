"""
Call the deployed image-classification endpoint.

The predictor takes raw image bytes and returns::

    {"predictions": [{"class_id": int, "class_name": str, "score": float}, ...],
     "image_size": {"width": int, "height": int},
     "device": str, "elapsed_ms": float}

Two ways to send an image (both land as raw bytes in ``predict(data)``):
  * a local file  → ``client.predict_file(path)``       (Content-Type guessed)
  * a remote URL  → ``client.predict_url(url)``          (server fetches it)

Run (from the job folder)
-------------------------
    # submit.py prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY — add them to .env, then:
    python sdk_utils/predict.py                          # classifies a sample image
    python sdk_utils/predict.py ./cat.jpg
    python sdk_utils/predict.py https://example.com/dog.jpg ./bird.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")   # .env lives in the job folder, next to inference.yaml

# A stable public sample image (golden retriever) — good for the ImageNet fallback.
DEFAULT_IMAGES = ["https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"]


def _collect_images() -> list[str]:
    if len(sys.argv) > 1:
        return list(sys.argv[1:])
    return DEFAULT_IMAGES


def _classify(client: InferenceClient, image: str) -> dict:
    if image.startswith("http://") or image.startswith("https://"):
        return client.predict_url(image)
    return client.predict_file(image)


def _print_result(image: str, result: dict) -> None:
    label = image if len(image) <= 60 else "…" + image[-57:]
    if "error" in result:
        print(f"[predict] {label} ✗ {result['error']}")
        return
    size = result.get("image_size", {})
    print(
        f"[predict] {label}  "
        f"({size.get('width', '?')}×{size.get('height', '?')}, "
        f"{result.get('elapsed_ms', 0):.0f} ms on {result.get('device', '?')})"
    )
    for p in result.get("predictions", []):
        print(f"        {p.get('score', 0):6.1%}  {p.get('class_name')}  (id={p.get('class_id')})")


def main() -> None:
    client = InferenceClient.from_env(timeout=120.0)
    images = _collect_images()
    for image in images:
        _print_result(image, _classify(client, image))
        print()
    client.close()


if __name__ == "__main__":
    main()
