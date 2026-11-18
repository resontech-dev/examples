"""
Transcribe audio with the deployed Parakeet TDT v3 endpoint: file in, text
with timestamps out.

Run
---
    pip install -U "resontech>=0.2.1" python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                       # sample_data/hello_reson.wav
    python predict.py ./call.mp3            # WAV / FLAC / OGG / MP3, up to 30 MB
    python predict.py ./call.wav --json     # the whole response
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"
# The central proxy takes 32 MB per request: an hour of 16 kHz WAV is ~115 MB,
# an hour of 64 kbit/s MP3 ~29 MB.
MAX_BYTES = 30 * 1024 * 1024


def stamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    audio = Path(args[0]) if args else DEFAULT_AUDIO
    if not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")
    size = audio.stat().st_size
    if size > MAX_BYTES:
        sys.exit(
            f"[predict] {audio.name} is {size / 2**20:.0f} MB; the proxy takes 32 MB per request. "
            "Convert it to MP3 (ffmpeg -i in.wav -ac 1 -b:a 64k out.mp3) or cut it."
        )

    # An hour of audio takes well under a minute on a GPU; the timeout covers a cold replica.
    client = InferenceClient.from_env(timeout=600.0)
    print(f"[predict] {audio} ({size / 1024:.0f} KB)")
    # Raw bytes as application/octet-stream: the predictor reads the file itself.
    result = client.predict(audio.read_bytes())
    client.close()
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    note = " (local attention)" if result.get("local_attention") else ""
    print(f"[predict] {result.get('duration_s')} s of audio in {result.get('elapsed_ms', 0):.0f} ms{note}")
    print("─" * 72)
    segments = result.get("segments") or []
    if segments:
        for s in segments:
            print(f"[{stamp(s['start'])}] {s['text']}")
    else:
        print(result.get("text", ""))


if __name__ == "__main__":
    main()
