"""
Call the deployed SAM 2.1 segmentation endpoint.

Sends an image + point prompt as JSON (``predict_json`` → octet-stream —
the platform's JSON envelope would drop the prompt fields otherwise) and
saves the returned masks as PNG files next to this script.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                                # sample shapes.png, center-ish point
    python predict.py ./photo.jpg 420 310            # your image, point at (420, 310)
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_IMAGE = HERE / "sample_data" / "shapes.png"
DEFAULT_POINT = (160, 120)   # center of the circle in shapes.png


def main() -> None:
    args = sys.argv[1:]
    image = Path(args[0]) if args else DEFAULT_IMAGE
    point = (int(args[1]), int(args[2])) if len(args) >= 3 else DEFAULT_POINT
    if not image.is_file():
        sys.exit(f"[predict] image not found: {image}")

    client = InferenceClient.from_env(timeout=300.0)
    print(f"[predict] image: {image}  point: {point}")

    result = client.predict_json({
        "image_b64": base64.b64encode(image.read_bytes()).decode("ascii"),
        "points": [list(point)],
        "labels": [1],
        "multimask": True,
    })
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    size = result.get("image_size", {})
    print(f"[predict] {len(result.get('masks', []))} mask(s) for "
          f"{size.get('width')}×{size.get('height')} image "
          f"({result.get('elapsed_ms', 0):.0f} ms on {result.get('device', '?')})")

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    for i, mask in enumerate(result.get("masks", [])):
        path = out_dir / f"mask_{i}_score{mask['score']:.2f}.png"
        path.write_bytes(base64.b64decode(mask["png_b64"]))
        print(f"    #{i}: score={mask['score']:.3f} area={mask['area']}px "
              f"bbox={mask['bbox']} → {path.relative_to(HERE)}")

    client.close()


if __name__ == "__main__":
    main()
