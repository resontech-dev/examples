"""
Transcribe audio with the deployed Nemotron 3.5 ASR Streaming endpoint: file
in, text and its detected language out.

Run
---
    pip install -U "resontech>=0.2.1" python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                              # sample_data/hello_reson.wav
    python predict.py ./call.mp3                   # the model detects the language
    python predict.py ./call.mp3 --language uk-UA  # pin it (see README: VERIFY)
    python predict.py ./call.mp3 --json            # the whole response
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"
# The central proxy takes 32 MB per request; base64 (with --language) adds a third.
MAX_BYTES = 30 * 1024 * 1024


def take_option(args: list[str], name: str) -> tuple[list[str], str | None]:
    if name not in args:
        return args, None
    i = args.index(name)
    if i + 1 >= len(args):
        sys.exit(f"[predict] {name} needs a value, e.g. {name} uk-UA")
    return args[:i] + args[i + 2:], args[i + 1]


def main() -> None:
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    args, language = take_option(args, "--language")
    audio = Path(args[0]) if args else DEFAULT_AUDIO
    if not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")
    raw = audio.read_bytes()
    limit = MAX_BYTES * 3 // 4 if language else MAX_BYTES
    if len(raw) > limit:
        sys.exit(
            f"[predict] {audio.name} is {len(raw) / 2**20:.0f} MB, over the {limit / 2**20:.0f} MB this request can carry. "
            "Convert it to MP3 (ffmpeg -i in.wav -ac 1 -b:a 64k out.mp3) or cut it."
        )

    client = InferenceClient.from_env(timeout=600.0)
    print(f"[predict] {audio} ({len(raw) / 1024:.0f} KB), language: {language or 'detected by the model'}")
    if language:
        result = client.predict_json({"audio_b64": base64.b64encode(raw).decode("ascii"), "language": language})
    else:
        result = client.predict(raw)   # raw bytes as application/octet-stream
    client.close()
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"[predict] {result.get('duration_s')} s of audio in {result.get('elapsed_ms', 0):.0f} ms, "
        f"language {result.get('language') or '?'}, context {result.get('att_context_size')}"
    )
    print("─" * 72)
    print(result.get("text", ""))


if __name__ == "__main__":
    main()
