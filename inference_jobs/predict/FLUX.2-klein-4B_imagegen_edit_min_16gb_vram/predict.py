"""
Call the deployed FLUX.2 [klein] 4B endpoint: prompt in, image out; or
image(s) + instruction in, edited image out. Saves to out/.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py "a ceramic coffee cup on a wooden desk, morning light, product photo"
    python predict.py --n 4 --size 1344x768 "poster: 'RESON STUDIO' in bold neon letters, night city"
    python predict.py --image screen.png "make the header dark navy and the main button orange"
    python predict.py --image person.jpg --image room.jpg "put the person from image 1 on the sofa in image 2"
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_PROMPT = "a lighthouse on a rocky coast at dawn, waves, seagulls, cinematic, warm light"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--image", type=Path, action="append", default=[],
                    help="reference / image to edit; repeat for up to 4")
    ap.add_argument("--size", help="WxH, multiples of 16 (default 1024x1024, or the image's aspect)")
    ap.add_argument("--n", type=int, default=1, help="images per request, 1-4")
    ap.add_argument("--steps", type=int, help="default 4")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--format", choices=["png", "jpeg", "webp"], default="png")
    args = ap.parse_args()

    payload: dict = {"prompt": args.prompt, "num_images": args.n, "format": args.format}
    if args.size:
        try:
            payload["width"], payload["height"] = (int(v) for v in args.size.lower().split("x"))
        except ValueError:
            sys.exit("[predict] --size wants WxH, e.g. --size 1344x768")
    for key, value in (("steps", args.steps), ("seed", args.seed)):
        if value is not None:
            payload[key] = value
    if args.image:
        payload["images_b64"] = [base64.b64encode(p.read_bytes()).decode("ascii") for p in args.image]

    mode = f"edit ({len(args.image)} image(s))" if args.image else "text-to-image"
    print(f"[predict] {mode}: {args.prompt!r}")
    # The platform cuts a predictor request off after 300 s; this only has to outlast that.
    with InferenceClient.from_env(timeout=330.0) as client:
        result = client.predict_json(payload)

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    ext = "jpg" if result["format"] == "jpeg" else result["format"]
    for seed, b64 in zip(result["seeds"], result["images_b64"]):
        path = out / f"klein_{seed}.{ext}"
        path.write_bytes(base64.b64decode(b64))
        print(f"[predict] → {path.relative_to(HERE)}")
    print(f"[predict] {result['mode']}, {result['width']}×{result['height']}, {result['steps']} steps, "
          f"seeds={result['seeds']}, {result['elapsed_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
