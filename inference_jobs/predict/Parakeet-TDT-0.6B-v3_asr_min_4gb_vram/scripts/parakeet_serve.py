"""Parakeet TDT 0.6B v3: audio in, text with timestamps out (NeMo, predictor mode).

Request, either:

  - the audio file itself as the body (WAV, FLAC, OGG or MP3), e.g.
    ``client.predict_file("call.wav")``;
  - JSON via ``client.predict_json``:

        {"audio_b64": "<base64 of an audio file>",
         "timestamps": true}                      # optional, default from init_args

Response:

    {"text": "Добрий день, це Іван…",
     "segments": [{"start": 0.0, "end": 2.4, "text": "Добрий день,"}, …],
     "duration_s": 12.3, "local_attention": false,
     "model_id": "nvidia/parakeet-tdt-0.6b-v3", "elapsed_ms": 812.0}

The model finds the language by itself (25 European languages, Ukrainian
and Russian included) and writes punctuation and capitals.

Long audio: full attention costs memory that grows with the square of the
length (NVIDIA: about 24 minutes per request on an 80 GB A100). Above
``local_attention_after_s`` the encoder switches to local attention, which
NVIDIA rates for up to 3 hours, and back to full attention for short clips.

Audio is decoded with soundfile (libsndfile ships inside the wheel), so the
worker needs no FFmpeg. M4A/AAC is not decoded: convert it first.

VALIDATE BEFORE LISTING: written, not booted. VERIFY list in README.md.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
import time
from math import gcd
from typing import Any

import numpy as np


SAMPLE_RATE = 16_000


class ParakeetASR:
    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        local_attention_after_s: float = 600.0,
        timestamps: bool = True,
    ) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        self.model_id = model_id
        self.local_after_s = float(local_attention_after_s)
        self.timestamps = bool(timestamps)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ~2.5 GB .nemo checkpoint from HF Hub on first boot.
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location=self.device)
        self.model.eval()
        self._local = False
        # One transcription at a time per replica: the GPU is the bottleneck
        # anyway, and the attention mode is switched per request.
        self._lock = threading.Lock()

    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        audio_bytes, opts = _parse_request(data)
        samples = _decode_16k_mono(audio_bytes)
        duration_s = len(samples) / SAMPLE_RATE
        if duration_s < 0.1:
            raise ValueError("the audio is empty or shorter than 0.1 s")
        want_ts = bool(opts.get("timestamps", self.timestamps))

        with self._lock, tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            _write_wav(tmp.name, samples)
            self._set_local_attention(duration_s > self.local_after_s)
            out = self.model.transcribe([tmp.name], batch_size=1, timestamps=want_ts, verbose=False)
            local = self._local

        hyp = _first_hypothesis(out)
        return {
            "text": (getattr(hyp, "text", None) or str(hyp)).strip(),
            "segments": _segments(hyp) if want_ts else [],
            "duration_s": round(duration_s, 2),
            "local_attention": local,
            "model_id": self.model_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def _set_local_attention(self, on: bool) -> None:
        if on == self._local:
            return
        if on:
            self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        else:
            # The checkpoint's own setting: full relative-position attention.
            self.model.change_attention_model(self_attention_model="rel_pos", att_context_size=[-1, -1])
        self._local = on

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model_id": self.model_id, "device": self.device, "local_attention": self._local}


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


def _write_wav(path: str, samples: np.ndarray) -> None:
    import soundfile as sf

    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def _first_hypothesis(out: Any) -> Any:
    # NeMo returns a list of hypotheses, or (best, all) for some decoders.
    if isinstance(out, tuple):
        out = out[0]
    return out[0]


def _segments(hyp: Any) -> list[dict[str, Any]]:
    stamps = (getattr(hyp, "timestamp", None) or {}).get("segment") or []
    return [
        {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": str(s["segment"]).strip()}
        for s in stamps
    ]
