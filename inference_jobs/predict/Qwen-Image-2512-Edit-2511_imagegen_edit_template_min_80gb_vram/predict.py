"""
Call the deployed Qwen-Image-2512 + Edit-2511 endpoint: prompt in, image out;
or image(s) + instruction in, edited image out. Saves to out/.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py 'poster: "GRAND OPENING Saturday 10:00", art-deco style, gold on black'
    python predict.py --size 1664x928 "a ceramic coffee cup on a wooden desk, morning light, product photo"
    python predict.py --image screen.png "make the header dark navy and the main button orange, keep everything else"
    python predict.py --image bear1.png --image bear2.png "the bear from image 1 and the bear from image 2 shake hands in a park"
    python predict.py --cfg 1 --steps 30 "a quick draft of a red fox"    # CFG off: about 3× faster, weaker
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

DEFAULT_PROMPT = 'a bookshop window at dusk, a hand-painted sign reads "OPEN LATE", warm light, rain on the glass'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--image", type=Path, action="append", default=[],
                    help="image to edit / reference; repeat for up to 3")
    ap.add_argument("--size", help="WxH, multiples of 16 (default 1328x1328, or the image's aspect at ~1 MP)")
    ap.add_argument("--n", type=int, default=1, help="images per request, 1-2")
    ap.add_argument("--steps", type=int, help="default 50 (text-to-image) / 40 (edit)")
    ap.add_argument("--cfg", type=float, help="true_cfg_scale, default 4; 1 turns CFG off")
    ap.add_argument("--negative", help="negative prompt (default: the model card's)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--format", choices=["png", "jpeg", "webp"], default="png")
    args = ap.parse_args()

    payload: dict = {"prompt": args.prompt, "num_images": args.n, "format": args.format}
    if args.size:
        try:
            payload["width"], payload["height"] = (int(v) for v in args.size.lower().split("x"))
        except ValueError:
            sys.exit("[predict] --size wants WxH, e.g. --size 1664x928")
    for key, value in (("steps", args.steps), ("seed", args.seed), ("true_cfg_scale", args.cfg),
                       ("negative_prompt", args.negative)):
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
        path = out / f"qwen_{result['mode']}_{seed}.{ext}"
        path.write_bytes(base64.b64decode(b64))
        print(f"[predict] → {path.relative_to(HERE)}")
    print(f"[predict] {result['mode']} ({result['model_id']}), {result['width']}×{result['height']}, "
          f"{result['steps']} steps, seeds={result['seeds']}, {result['elapsed_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
