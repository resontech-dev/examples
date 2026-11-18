"""pyannote Community-1: who spoke when (speaker diarization, predictor mode).

Request, either:

  - the audio file itself as the body (WAV, FLAC, OGG or MP3);
  - JSON via ``client.predict_json``:

        {"audio_b64": "<base64 of an audio file>",
         "num_speakers": 2,                       # optional: exact count
         "min_speakers": 2, "max_speakers": 5}    # optional: a range

Response:

    {"speakers": ["SPEAKER_00", "SPEAKER_01"],
     "turns": [{"speaker": "SPEAKER_00", "start": 0.52, "end": 4.1}, …],
     "exclusive_turns": [...],      # one speaker at a time: ready to cut for transcription
     "duration_s": 61.2, "elapsed_ms": 2300.0}

``turns`` may overlap (two people talking at once); ``exclusive_turns``
never do, which is what a transcription per speaker needs.

Weights: the pipeline is gated on Hugging Face (accept its conditions once).
``pack_weights.py`` downloads it on your machine with your token and packs it
into ``model/speaker-diarization-community-1.tar.gz``; submit.py ships that
file, the platform mounts it and calls ``load_weights``. No token ever
reaches the job.

Audio is decoded with soundfile and handed to pyannote as a waveform, so
pyannote never calls torchcodec/FFmpeg (the worker image has no FFmpeg).

VALIDATE BEFORE LISTING: written, not booted. VERIFY list in README.md.
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
import tempfile
import threading
import time
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 16_000


class Diarizer:
    def __init__(self, num_speakers: int | None = None, min_speakers: int | None = None,
                 max_speakers: int | None = None) -> None:
        import torch

        self.defaults = {k: v for k, v in {
            "num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers,
        }.items() if v}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._lock = threading.Lock()

    def load_weights(self, weights_path: str) -> None:
        """Unpack the pipeline folder that pack_weights.py made and load it offline."""
        from pyannote.audio import Pipeline

        target = Path(tempfile.mkdtemp(prefix="pyannote-"))
        with tarfile.open(weights_path) as tar:
            tar.extractall(target, filter="data")
        configs = sorted(target.rglob("config.yaml"), key=lambda p: len(p.parts))
        if not configs:
            raise RuntimeError(f"{weights_path} has no config.yaml: pack it with pack_weights.py")
        self.pipeline = Pipeline.from_pretrained(str(configs[0].parent))
        self.pipeline.to(self.device)

    def predict(self, data: bytes) -> dict[str, Any]:
        import torch

        if self.pipeline is None:
            raise RuntimeError(
                "the pipeline is not loaded: inference.yaml needs model.path and submit.py "
                "must ship model/speaker-diarization-community-1.tar.gz (see pack_weights.py)"
            )
        t0 = time.perf_counter()
        audio_bytes, opts = _parse_request(data)
        samples = _decode_16k_mono(audio_bytes)
        duration_s = len(samples) / SAMPLE_RATE
        if duration_s < 0.5:
            raise ValueError("the audio is empty or shorter than 0.5 s")
        counts = dict(self.defaults)
        counts.update({k: int(opts[k]) for k in ("num_speakers", "min_speakers", "max_speakers") if opts.get(k)})

        audio = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": SAMPLE_RATE}
        with self._lock:
            out = self.pipeline(audio, **counts)

        turns = _turns(out.speaker_diarization)
        exclusive = _turns(getattr(out, "exclusive_speaker_diarization", None) or out.speaker_diarization)
        return {
            "speakers": sorted({t["speaker"] for t in turns}),
            "turns": turns,
            "exclusive_turns": exclusive,
            "duration_s": round(duration_s, 2),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok" if self.pipeline is not None else "no weights", "device": str(self.device)}


def _turns(annotation: Any) -> list[dict[str, Any]]:
    return [
        {"speaker": str(speaker), "start": round(float(turn.start), 3), "end": round(float(turn.end), 3)}
        for turn, speaker in annotation
    ]


def _parse_request(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """The audio bytes and the options: a JSON envelope, or the audio itself."""
    if data[:1] == b"{":
        try:
            req = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            req = None
        if isinstance(req, dict):
            b64 = req.get("audio_b64") or req.get("data_b64")   # data_b64: client.predict_b64
            if not b64:
                raise ValueError("a JSON body needs 'audio_b64': the base64 of an audio file")
            return base64.b64decode(b64), req
    return data, {}


def _decode_16k_mono(raw: bytes) -> np.ndarray:
    import soundfile as sf

    try:
        audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"cannot decode the audio ({exc}); send WAV, FLAC, OGG or MP3") from exc
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(int(rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g)
    return mono.astype(np.float32)
