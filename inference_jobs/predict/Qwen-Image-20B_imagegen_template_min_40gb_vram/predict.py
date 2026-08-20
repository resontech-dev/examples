"""
Call the deployed Qwen-Image endpoint — prompt in, PNG out. (TEMPLATE)

Same client contract as the FLUX sibling — swap the endpoint, keep the code.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py 'poster: "GRAND OPENING — Saturday 10:00", art-deco style'
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

# Text rendering is this model's superpower — default prompt shows it off.
DEFAULT_PROMPT = (
    'A minimalist conference poster with the exact text "RESON — PRIVATE '
    'INFERENCE DAY" in bold sans-serif, teal on off-white, subtle grid'
)


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT

    # 50-step diffusion on a 20B MMDiT — generous timeout.
    client = InferenceClient.from_env(timeout=900.0)
    print(f"[predict] prompt: {prompt!r}")

    result = client.predict_json({"prompt": prompt})
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"qwen_image_{int(time.time())}.png"
    path.write_bytes(base64.b64decode(result["image_b64"]))
    print(f"[predict] {result['width']}×{result['height']} in "
          f"{result.get('elapsed_ms', 0):.0f} ms ({result.get('steps')} steps)")
    print(f"[predict] → {path.relative_to(HERE)}")

    client.close()


if __name__ == "__main__":
    main()
