"""
Call the deployed Orpheus-TTS endpoint — text in, WAV out.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py "Welcome to Reson. <sigh> Deploying models used to be hard."
    python predict.py --voice leo "Short status update: all systems nominal."
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_TEXT = "Hello! This voice is generated on your own hardware, and it never left the building."


def main() -> None:
    args = sys.argv[1:]
    voice = None
    if "--voice" in args:
        i = args.index("--voice")
        try:
            voice = args[i + 1]
        except IndexError:
            sys.exit("[predict] --voice needs a name (tara/leah/jess/leo/dan/mia/zac/zoe)")
        args = args[:i] + args[i + 2:]
    text = args[0] if args else DEFAULT_TEXT

    # TTS generation is slow (LM decode ≈ realtime-ish) — generous timeout.
    client = InferenceClient.from_env(timeout=600.0)
    payload = {"text": text}
    if voice:
        payload["voice"] = voice
    print(f"[predict] synthesizing ({voice or 'default voice'}): {text!r}")

    result = client.predict_json(payload)
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"tts_{result.get('voice', 'voice')}.wav"
    path.write_bytes(base64.b64decode(result["audio_b64"]))
    print(f"[predict] {result.get('duration_s', '?')} s of audio "
          f"({result.get('sample_rate')} Hz) in {result.get('elapsed_ms', 0):.0f} ms")
    print(f"[predict] → {path.relative_to(HERE)}   (afplay/open it to listen)")

    client.close()


if __name__ == "__main__":
    main()
