"""
Call the deployed FLUX.1-schnell endpoint — prompt in, PNG out.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py "product shot of a ceramic mug on slate, softbox lighting"
    python predict.py --size 1280x768 --seed 7 "isometric game asset, watermill"
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_PROMPT = "a lighthouse on a rocky coast at dawn, oil painting, warm light"


def main() -> None:
    args = sys.argv[1:]
    width = height = 1024
    seed = None
    if "--size" in args:
        i = args.index("--size")
        try:
            width, height = (int(v) for v in args[i + 1].lower().split("x"))
        except (IndexError, ValueError):
            sys.exit("[predict] --size wants WxH, e.g. --size 1280x768")
        args = args[:i] + args[i + 2:]
    if "--seed" in args:
        i = args.index("--seed")
        try:
            seed = int(args[i + 1])
        except (IndexError, ValueError):
            sys.exit("[predict] --seed wants an integer")
        args = args[:i] + args[i + 2:]
    prompt = args[0] if args else DEFAULT_PROMPT

    client = InferenceClient.from_env(timeout=600.0)
    print(f"[predict] {width}×{height}, prompt: {prompt!r}")

    payload = {"prompt": prompt, "width": width, "height": height}
    if seed is not None:
        payload["seed"] = seed
    result = client.predict_json(payload)
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"flux_{int(time.time())}.png"
    path.write_bytes(base64.b64decode(result["image_b64"]))
    print(f"[predict] {result['width']}×{result['height']} in "
          f"{result.get('elapsed_ms', 0):.0f} ms ({result.get('steps')} steps)"
          + (f", seed={result.get('seed')}" if result.get("seed") is not None else ""))
    print(f"[predict] → {path.relative_to(HERE)}")

    client.close()


if __name__ == "__main__":
    main()
