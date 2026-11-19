"""
Call the deployed LTX-2.3 endpoint: prompt (and optionally an image) in, MP4 with sound out.

Each request waits up to 20 s and the script keeps polling until the clip
is done, then saves it to out/. Describe the sound in the prompt as well:
the audio track follows it.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py "rain on a tin roof at night, a dog barks in the distance, slow dolly in"
    python predict.py --image photo.jpg "she turns to the camera and says: 'we open at nine'"
    python predict.py --frames 241 "a storm over the sea, thunder rolls, waves crash"   # 10 s
    python predict.py --status <id>        # pick up a clip that is still rendering
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

DEFAULT_PROMPT = (
    "a violet glow rises under dark waves at night, the camera pushes in slowly, "
    "lightning flickers on the horizon; deep rumbling thunder and crashing surf"
)
WAIT_S = 20    # per request, so the start returns the id fast; the job caps wait_s at 280 (300 s per request)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--image", type=Path, help="start the clip from this image (image-to-video)")
    ap.add_argument("--strength", type=float, help="how strictly to keep the image, 0–1 (default 1)")
    ap.add_argument("--size", help="WxH, multiples of 64 (default 1280x704, or the image's aspect)")
    ap.add_argument("--frames", type=int, help="8k+1: 97 ≈ 4 s, 121 ≈ 5 s, 241 ≈ 10 s (default 121)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--status", metavar="ID", help="poll an existing job instead of starting one")
    args = ap.parse_args()

    client = InferenceClient.from_env(timeout=WAIT_S + 30.0)

    if args.status:
        result = client.predict_json({"action": "status", "id": args.status, "wait_s": WAIT_S})
    else:
        payload = {"prompt": args.prompt, "wait_s": WAIT_S}
        if args.size:
            try:
                payload["width"], payload["height"] = (int(v) for v in args.size.lower().split("x"))
            except ValueError:
                sys.exit("[predict] --size wants WxH, e.g. --size 1280x704")
        for key, value in (("num_frames", args.frames), ("seed", args.seed), ("image_strength", args.strength)):
            if value is not None:
                payload[key] = value
        if args.image:
            payload["image_b64"] = base64.b64encode(args.image.read_bytes()).decode("ascii")
        print(f"[predict] {'image-to-video' if args.image else 'text-to-video'}: {args.prompt!r}")
        result = client.predict_json(payload)

    while result.get("status") in ("queued", "in_progress"):
        if result["status"] == "queued":
            print(f"[predict] {result['id']}: queued, {result.get('queue_position', 0)} ahead")
        else:
            print(f"[predict] {result['id']}: rendering for {result.get('running_s', 0)} s "
                  f"(Ctrl-C is safe, resume with --status {result['id']})")
        result = client.predict_json({"action": "status", "id": result["id"], "wait_s": WAIT_S})

    if "error" in result:
        sys.exit(f"[predict] failed: {result['error']}")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"ltx_{result['id']}.mp4"
    path.write_bytes(base64.b64decode(result["video_b64"]))
    print(f"[predict] {result['width']}×{result['height']}, {result['num_frames']} frames "
          f"({result['seconds']} s at {result['fps']:g} fps, with audio), seed={result['seed']}, "
          f"rendered in {result['elapsed_ms'] / 1000:.0f} s")
    print(f"[predict] → {path.relative_to(HERE)}")
    client.close()


if __name__ == "__main__":
    main()
