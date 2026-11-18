"""
Stream audio to the deployed Voxtral Realtime endpoint and print the text as
it arrives (vLLM realtime API over WebSocket), or send a whole file to the
transcription route.

Run
---
    pip install python-dotenv websockets soundfile numpy scipy openai
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                         # sample_data/hello_reson.wav, streamed as fast as possible
    python predict.py ./call.wav --pace       # streamed at speaking speed, like a live microphone
    python predict.py ./call.wav --transcribe # plan B: POST /v1/audio/transcriptions, whole file

The realtime route is the one flagged VALIDATE in README.md: a failed
WebSocket handshake (404 / 426) means the router or the proxy does not pass it.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from math import gcd
from pathlib import Path

from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"
SAMPLE_RATE = 16_000
CHUNK_BYTES = 4096                    # PCM16 per append, as in vLLM's example client


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _base(raw: str) -> str:
    base = raw.rstrip("/")
    return base[: -len("/predict")] if base.endswith("/predict") else base


def pcm16_16k(path: Path) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError:
        sys.exit("[predict] pip install soundfile numpy scipy")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(int(rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g)
    return (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()


async def realtime(audio: Path, pace: bool) -> None:
    try:
        import websockets
    except ImportError:
        sys.exit("[predict] pip install websockets")
    key = _required("RESON_INFERENCE_API_KEY")
    url = _base(_required("RESON_INFERENCE_URL")).replace("https://", "wss://").replace("http://", "ws://") + "/v1/realtime"
    model = os.getenv("MODEL_ID", "voxtral")
    pcm = pcm16_16k(audio)
    headers = {"Authorization": f"Bearer {key}", "X-API-Key": key}
    print(f"[predict] {audio} ({len(pcm) / 2 / SAMPLE_RATE:.1f} s) -> {url}")
    print("─" * 72)

    try:
        connect = websockets.connect(url, additional_headers=headers, max_size=None)   # websockets >= 14
    except TypeError:
        connect = websockets.connect(url, extra_headers=headers, max_size=None)        # older releases
    async with connect as ws:
        first = json.loads(await ws.recv())
        if first.get("type") != "session.created":
            sys.exit(f"[predict] expected session.created, got: {first}")
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def send_audio() -> None:
            for i in range(0, len(pcm), CHUNK_BYTES):
                chunk = pcm[i:i + CHUNK_BYTES]
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(chunk).decode("ascii")}))
                if pace:
                    await asyncio.sleep(len(chunk) / 2 / SAMPLE_RATE)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        sender = asyncio.create_task(send_audio())
        async for message in ws:
            event = json.loads(message)
            kind = event.get("type")
            if kind == "transcription.delta":
                print(event.get("delta", ""), end="", flush=True)
            elif kind == "transcription.done":
                print()
                break
            elif kind == "error":
                sys.exit(f"\n[predict] server error: {event.get('error')}")
        await sender


def transcribe(audio: Path) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[predict] pip install openai")
    key = _required("RESON_INFERENCE_API_KEY")
    client = OpenAI(base_url=_base(_required("RESON_INFERENCE_URL")) + "/v1", api_key=key,
                    default_headers={"X-API-Key": key}, timeout=300.0)
    with audio.open("rb") as fh:
        result = client.audio.transcriptions.create(model=os.getenv("MODEL_ID", "voxtral"), file=fh)
    print(result.text)


def main() -> None:
    args = sys.argv[1:]
    pace = "--pace" in args
    whole = "--transcribe" in args
    args = [a for a in args if a not in ("--pace", "--transcribe")]
    audio = Path(args[0]) if args else DEFAULT_AUDIO
    if not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")
    if whole:
        transcribe(audio)
    else:
        asyncio.run(realtime(audio, pace))


if __name__ == "__main__":
    main()
