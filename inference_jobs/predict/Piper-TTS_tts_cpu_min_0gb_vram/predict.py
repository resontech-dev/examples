"""
Call the deployed Piper TTS endpoint — text in, WAV out.

Run
---
    pip install python-dotenv resontech
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py
    python predict.py "Ваше замовлення №10452 буде доставлено завтра до 14:00."
    python predict.py --voice en_US-lessac-medium "Your order ships tomorrow."
    python predict.py --voice it_IT-paola-medium "Il suo ordine arriva domani."
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_TEXT = "Добрий день! Це голос, згенерований на вашому власному сервері. Дані нікуди не передавалися."


def main() -> None:
    args = sys.argv[1:]
    voice = None
    if "--voice" in args:
        i = args.index("--voice")
        try:
            voice = args[i + 1]
        except IndexError:
            sys.exit("[predict] --voice needs a name, e.g. uk_UA-ukrainian_tts-medium")
        args = args[:i] + args[i + 2:]
    text = args[0] if args else DEFAULT_TEXT

    client = InferenceClient.from_env(timeout=120.0)
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
