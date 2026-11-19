"""
Call the deployed Wan2.2-TI2V-5B endpoint: prompt (and optionally an image) in, MP4 out.

A clip takes minutes; each request waits up to 20 s and the script keeps
polling until the clip is done, then saves it to out/.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py "a red fox trots through fresh snow, low tracking shot, golden hour"
    python predict.py --image photo.jpg "the camera slowly pushes in, leaves drift in the wind"
    python predict.py --frames 49 --steps 30 "draft: waves crash on black rocks"   # ~2 s, faster
    python predict.py --size 704x1280 "vertical: neon street in the rain"
    python predict.py --status <id>        # pick up a clip that is still rendering
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_PROMPT = (
    "a lighthouse on a rocky coast at dawn, waves crash against the rocks, "
    "seagulls circle, slow aerial orbit, cinematic, warm light"
)
WAIT_S = 20    # per request, so the start returns the id fast; the job caps wait_s at 280 (300 s per request)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--image", type=Path, help="animate this image (image-to-video)")
    ap.add_argument("--size", default="1280x704", help="WxH, multiples of 32 (default 1280x704)")
    ap.add_argument("--frames", type=int, help="4k+1: 49 ≈ 2 s, 81 ≈ 3.4 s, 121 ≈ 5 s (default 121)")
    ap.add_argument("--steps", type=int, help="denoising steps (default 50)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--status", metavar="ID", help="poll an existing job instead of starting one")
    args = ap.parse_args()

    client = InferenceClient.from_env(timeout=WAIT_S + 30.0)

    if args.status:
        result = client.predict_json({"action": "status", "id": args.status, "wait_s": WAIT_S})
    else:
        try:
            width, height = (int(v) for v in args.size.lower().split("x"))
        except ValueError:
            sys.exit("[predict] --size wants WxH, e.g. --size 1280x704")
        payload = {"prompt": args.prompt, "width": width, "height": height, "wait_s": WAIT_S}
        for key, value in (("num_frames", args.frames), ("steps", args.steps), ("seed", args.seed)):
            if value is not None:
                payload[key] = value
        if args.image:
            payload["image_b64"] = base64.b64encode(args.image.read_bytes()).decode("ascii")
        print(f"[predict] {'image-to-video' if args.image else 'text-to-video'}: {args.prompt!r}")
        result = client.predict_json(payload)

    t0 = time.time()
    while result.get("status") in ("queued", "in_progress"):
        _report(result, time.time() - t0)
        result = client.predict_json({"action": "status", "id": result["id"], "wait_s": WAIT_S})

    if "error" in result:
        sys.exit(f"[predict] failed: {result['error']}")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"wan_{result['id']}.mp4"
    path.write_bytes(base64.b64decode(result["video_b64"]))
    print(f"[predict] {result['width']}×{result['height']}, {result['num_frames']} frames "
          f"({result['seconds']} s at {result['fps']} fps), {result['steps']} steps, "
          f"seed={result['seed']}, rendered in {result['elapsed_ms'] / 1000:.0f} s")
    print(f"[predict] → {path.relative_to(HERE)}")
    client.close()


def _report(result: dict, waited: float) -> None:
    if result["status"] == "queued":
        print(f"[predict] {result['id']}: queued, {result.get('queue_position', 0)} ahead")
    else:
        print(f"[predict] {result['id']}: {result.get('progress', 0) * 100:.0f} % "
              f"(waited {waited:.0f} s; Ctrl-C is safe, resume with --status {result['id']})")


if __name__ == "__main__":
    main()
