"""
Speech-to-text with the deployed Nemotron 3.5 ASR Streaming endpoint: live
from the microphone, a file streamed at speaking speed, or a whole file.

Run
---
    pip install -U "resontech>=0.2.1" python-dotenv numpy soundfile scipy
    pip install sounddevice                        # only for --mic
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py --mic --language uk-UA       # speak; Ctrl-C ends the session
    python predict.py --live ./call.wav --language uk-UA   # the file at speaking speed, like a microphone
    python predict.py --live ./call.wav --fast     # as fast as the endpoint answers
    python predict.py                              # whole file: sample_data/hello_reson.wav
    python predict.py ./call.mp3 --language uk-UA  # whole file, language pinned
    python predict.py ./call.mp3 --json            # whole file, the whole response
"""
from __future__ import annotations

import argparse
import base64
import json
import queue
import sys
import time
from math import gcd
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"
SAMPLE_RATE = 16_000
# The central proxy takes 32 MB per request; base64 (with --language) adds a third.
MAX_BYTES = 30 * 1024 * 1024


# ── live ────────────────────────────────────────────────────────────────────


class Live:
    """One live session: start, append PCM16 pieces, stop. Prints the text as it settles."""

    def __init__(self, client: InferenceClient, language: str | None) -> None:
        self.client = client
        body: dict = {"action": "start"}
        if language:
            body["language"] = language
        reply = self._call(body)
        self.session = reply["session"]
        self.frame_ms = reply["frame_ms"]
        self.rtts: list[float] = []
        two_pass = f", final text by {reply['final_model']}" if reply.get("final_model") else ""
        print(f"[live] session {self.session[:8]}…, frame {self.frame_ms} ms, language {reply.get('language')}{two_pass}")
        print("─" * 72)

    def append(self, pcm: bytes, final: bool = False) -> dict:
        t0 = time.perf_counter()
        reply = self._call({
            "action": "append", "session": self.session, "final": final,
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
        })
        self.rtts.append(time.perf_counter() - t0)
        self._show(reply)
        return reply

    def _call(self, body: dict) -> dict:
        reply = self.client.predict_json(body)
        if "error" in reply:
            err = reply["error"]
            sys.exit(f"\n[live] {err.get('message', err) if isinstance(err, dict) else err}")
        return reply

    @staticmethod
    def _show(reply: dict) -> None:
        if reply.get("final"):
            sys.stdout.write("\r\x1b[2K" + reply["text"] + "\n")
            return
        # One status line: the end of the settled text, then the words still being decided (dim).
        text, partial = reply["text"][-90:], reply["partial"][-60:]
        sys.stdout.write("\r\x1b[2K" + text + (" \x1b[2m" + partial + "\x1b[0m" if partial else ""))
        sys.stdout.flush()

    def summary(self) -> str:
        if not self.rtts:
            return ""
        ms = sorted(r * 1000 for r in self.rtts)
        return (f"[live] {len(ms)} requests, round trip median {ms[len(ms) // 2]:.0f} ms, "
                f"p90 {ms[int(len(ms) * 0.9)]:.0f} ms; text lags the voice by about one frame "
                f"({self.frame_ms} ms) plus the round trip")


def pcm16_16k(path: Path) -> bytes:
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(int(rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g)
    return (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()


def live_file(client: InferenceClient, audio: Path, language: str | None, fast: bool) -> None:
    pcm = pcm16_16k(audio)
    print(f"[live] {audio} ({len(pcm) / 2 / SAMPLE_RATE:.1f} s), {'as fast as possible' if fast else 'at speaking speed'}")
    live = Live(client, language)
    step = SAMPLE_RATE * 2 * live.frame_ms // 1000          # bytes of one frame
    t0 = time.monotonic()
    for start in range(0, len(pcm), step):
        piece = pcm[start:start + step]
        if not fast:
            # a microphone hands over a frame once it has been spoken
            wait = t0 + (start + len(piece)) / 2 / SAMPLE_RATE - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        live.append(piece, final=start + step >= len(pcm))
    print(live.summary())


def live_mic(client: InferenceClient, language: str | None) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("[live] pip install sounddevice (and on Linux: apt install libportaudio2)")
    live = Live(client, language)
    blocks: queue.Queue[bytes] = queue.Queue()

    def on_audio(indata, frames, t, status) -> None:   # runs in the audio thread
        blocks.put(bytes(indata))

    print("[live] listening… speak, Ctrl-C to finish")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                           blocksize=SAMPLE_RATE * live.frame_ms // 1000, callback=on_audio):
        try:
            while True:
                piece = blocks.get()
                while not blocks.empty():                # the endpoint was slow: send what piled up
                    piece += blocks.get_nowait()
                live.append(piece)
        except KeyboardInterrupt:
            pass
    rest = b""
    while not blocks.empty():
        rest += blocks.get_nowait()
    live.append(rest, final=True)
    print(live.summary())


# ── whole file ──────────────────────────────────────────────────────────────


def whole_file(client: InferenceClient, audio: Path, language: str | None, as_json: bool) -> None:
    raw = audio.read_bytes()
    limit = MAX_BYTES * 3 // 4 if language else MAX_BYTES
    if len(raw) > limit:
        sys.exit(
            f"[predict] {audio.name} is {len(raw) / 2**20:.0f} MB, over the {limit / 2**20:.0f} MB this request can carry. "
            "Convert it to MP3 (ffmpeg -i in.wav -ac 1 -b:a 64k out.mp3) or cut it."
        )
    print(f"[predict] {audio} ({len(raw) / 1024:.0f} KB), language: {language or 'detected by the model'}")
    if language:
        result = client.predict_json({"audio_b64": base64.b64encode(raw).decode("ascii"), "language": language})
    else:
        result = client.predict(raw)   # raw bytes as application/octet-stream
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", type=Path, help="audio file (default: sample_data/hello_reson.wav)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mic", action="store_true", help="live session from the microphone")
    mode.add_argument("--live", action="store_true", help="live session fed from the audio file")
    ap.add_argument("--fast", action="store_true", help="with --live: do not wait for real time")
    ap.add_argument("--language", help="e.g. uk-UA, en-US; default: the job's live_language / detected")
    ap.add_argument("--json", action="store_true", help="whole file: print the whole response")
    args = ap.parse_args()

    audio = args.audio or DEFAULT_AUDIO
    if not args.mic and not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")
    # Live requests are short; a whole file can take a while.
    with InferenceClient.from_env(timeout=30.0 if (args.mic or args.live) else 330.0) as client:
        if args.mic:
            live_mic(client, args.language)
        elif args.live:
            live_file(client, audio, args.language, args.fast)
        else:
            whole_file(client, audio, args.language, args.json)


if __name__ == "__main__":
    main()
