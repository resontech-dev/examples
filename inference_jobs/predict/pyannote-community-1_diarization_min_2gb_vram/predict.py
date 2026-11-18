"""
Who spoke when, with the deployed pyannote Community-1 endpoint; with
--with-whisper also what each speaker said, through a running Whisper job.

Run
---
    pip install -U "resontech>=0.2.1" python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py ./call.mp3                    # turns: [00:04] SPEAKER_01 (3.2 s)
    python predict.py ./call.mp3 --speakers 2       # the count is known
    python predict.py ./call.mp3 --with-whisper     # [00:04] SPEAKER_01: text
    python predict.py ./call.mp3 --with-whisper --language uk

--with-whisper also needs `pip install openai soundfile numpy` and WHISPER_URL +
WHISPER_API_KEY in .env (the endpoint and key of a Whisper job, e.g.
../../vllm/whisper-large-v3_asr_min_6gb_vram). The audio is cut on this
machine at the turns pyannote found; each piece goes to Whisper on its own.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"
MAX_BYTES = 30 * 1024 * 1024        # the central proxy takes 32 MB per request
PIECE_MAX_S = 28.0                  # Whisper hears 30 s at a time
JOIN_GAP_S = 1.0                    # turns of one speaker closer than this are sent together


def stamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def take_option(args: list[str], name: str) -> tuple[list[str], str | None]:
    if name not in args:
        return args, None
    i = args.index(name)
    if i + 1 >= len(args):
        sys.exit(f"[predict] {name} needs a value")
    return args[:i] + args[i + 2:], args[i + 1]


def pieces(turns: list[dict]) -> list[dict]:
    """Neighbouring turns of one speaker joined, then cut to Whisper-sized pieces."""
    joined: list[dict] = []
    for t in sorted(turns, key=lambda t: t["start"]):
        last = joined[-1] if joined else None
        if (last and last["speaker"] == t["speaker"] and t["start"] - last["end"] <= JOIN_GAP_S
                and t["end"] - last["start"] <= PIECE_MAX_S):
            last["end"] = t["end"]
        else:
            joined.append(dict(t))
    out: list[dict] = []
    for p in joined:
        start = p["start"]
        while p["end"] - start > PIECE_MAX_S:
            out.append({"speaker": p["speaker"], "start": start, "end": start + PIECE_MAX_S})
            start += PIECE_MAX_S
        out.append({"speaker": p["speaker"], "start": start, "end": p["end"]})
    return [p for p in out if p["end"] - p["start"] >= 0.3]


def with_whisper(audio: Path, turns: list[dict], language: str | None) -> None:
    try:
        import numpy  # noqa: F401  (soundfile needs it)
        import soundfile as sf
        from openai import OpenAI
    except ImportError:
        sys.exit("[predict] --with-whisper needs: pip install openai soundfile numpy")
    url, key = os.getenv("WHISPER_URL"), os.getenv("WHISPER_API_KEY")
    if not url or not key:
        sys.exit("[predict] --with-whisper needs WHISPER_URL and WHISPER_API_KEY in .env (a running Whisper job)")
    base = url.rstrip("/").removesuffix("/predict") + "/v1"
    client = OpenAI(base_url=base, api_key=key, default_headers={"X-API-Key": key}, timeout=300.0)
    model = os.getenv("WHISPER_MODEL") or client.models.list().data[0].id

    samples, rate = sf.read(str(audio), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    todo = pieces(turns)
    print(f"[predict] {len(todo)} pieces to {model} at {base}")
    print("─" * 72)
    for p in todo:
        buf = io.BytesIO()
        sf.write(buf, mono[int(p["start"] * rate):int(p["end"] * rate)], rate, format="WAV", subtype="PCM_16")
        kwargs = {"model": model, "file": ("turn.wav", buf.getvalue(), "audio/wav"), "response_format": "json"}
        if language:
            kwargs["language"] = language
        text = client.audio.transcriptions.create(**kwargs).text.strip()
        if text:
            print(f"[{stamp(p['start'])}] {p['speaker']}: {text}")


def main() -> None:
    args = sys.argv[1:]
    as_json = "--json" in args
    whisper = "--with-whisper" in args
    args = [a for a in args if a not in ("--json", "--with-whisper")]
    args, speakers = take_option(args, "--speakers")
    args, language = take_option(args, "--language")
    audio = Path(args[0]) if args else DEFAULT_AUDIO
    if not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")
    raw = audio.read_bytes()
    limit = MAX_BYTES * 3 // 4 if speakers else MAX_BYTES
    if len(raw) > limit:
        sys.exit(
            f"[predict] {audio.name} is {len(raw) / 2**20:.0f} MB, over the {limit / 2**20:.0f} MB this request can carry. "
            "Convert it to MP3 (ffmpeg -i in.wav -ac 1 -b:a 64k out.mp3) or cut it."
        )

    client = InferenceClient.from_env(timeout=600.0)
    print(f"[predict] {audio} ({len(raw) / 1024:.0f} KB)")
    if speakers:
        result = client.predict_json({"audio_b64": base64.b64encode(raw).decode("ascii"), "num_speakers": int(speakers)})
    else:
        result = client.predict(raw)   # raw bytes as application/octet-stream
    client.close()
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    turns = result.get("exclusive_turns") or result.get("turns") or []
    print(f"[predict] {len(result.get('speakers', []))} speakers, {len(turns)} turns in "
          f"{result.get('duration_s')} s of audio ({result.get('elapsed_ms', 0):.0f} ms)")
    if whisper:
        with_whisper(audio, turns, language)
        return
    print("─" * 72)
    for t in turns:
        print(f"[{stamp(t['start'])}] {t['speaker']} ({t['end'] - t['start']:.1f} s)")


if __name__ == "__main__":
    main()
