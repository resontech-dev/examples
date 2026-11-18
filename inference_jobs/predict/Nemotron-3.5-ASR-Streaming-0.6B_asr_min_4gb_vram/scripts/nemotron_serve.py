"""Nemotron 3.5 ASR Streaming 0.6B: audio in, text and its language out (NeMo, predictor mode).

A cache-aware streaming FastConformer + RNN-T model (NVIDIA, OpenMDW-1.1):
built to transcribe live audio in chunks of 80 ms to 1.12 s, in 40 language
locales, 19 of them "transcription-ready", Ukrainian (uk-UA) among them.

This predictor transcribes whole files (request/response); the platform has
no WebSocket route for a live session yet. What it shows: accuracy and speed
at the latency the model is set to (``att_context_size``), and the language
the model detected.

Request, either:

  - the audio file itself as the body (WAV, FLAC, OGG or MP3);
  - JSON via ``client.predict_json``:

        {"audio_b64": "<base64 of an audio file>",
         "language": "uk-UA",        # optional; default: the model detects it
         "timestamps": false}        # optional

Response:

    {"text": "Добрий день, це Іван…", "language": "uk-UA",
     "segments": [], "duration_s": 12.3, "att_context_size": [56, 13],
     "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b", "elapsed_ms": 640.0}

With automatic language detection the model appends a tag such as
``<uk-UA>`` to the text; it is cut off and returned as ``language``.

VALIDATE BEFORE LISTING: written, not booted. VERIFY list in README.md.
"""
from __future__ import annotations

import base64
import io
import json
import re
import tempfile
import threading
import time
from math import gcd
from typing import Any

import numpy as np


SAMPLE_RATE = 16_000
LANG_TAG = re.compile(r"\s*<([a-z]{2,3}-[A-Z]{2})>\s*")


class NemotronStreamingASR:
    def __init__(
        self,
        model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        att_context_size: list[int] | None = None,
        timestamps: bool = False,
    ) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        self.model_id = model_id
        self.timestamps = bool(timestamps)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location=self.device)
        self.model.eval()
        # [left, right] context in 80 ms frames; the right one sets the latency:
        # 0 -> 80 ms, 1 -> 160 ms, 3 -> 320 ms, 6 -> 560 ms, 13 -> 1.12 s (most accurate).
        self.att_context_size = list(att_context_size or [56, 13])
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None and hasattr(encoder, "set_default_att_context_size"):
            encoder.set_default_att_context_size(self.att_context_size)
        self._lock = threading.Lock()

    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        audio_bytes, opts = _parse_request(data)
        samples = _decode_16k_mono(audio_bytes)
        duration_s = len(samples) / SAMPLE_RATE
        if duration_s < 0.1:
            raise ValueError("the audio is empty or shorter than 0.1 s")
        want_ts = bool(opts.get("timestamps", self.timestamps))
        language = opts.get("language") or None

        kwargs: dict[str, Any] = {"batch_size": 1, "timestamps": want_ts, "verbose": False}
        if language:
            # The model card sets the language as `target_lang` in NeMo's
            # streaming script; transcribe() taking it is on the VERIFY list.
            kwargs["target_lang"] = language
        with self._lock, tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            _write_wav(tmp.name, samples)
            try:
                out = self.model.transcribe([tmp.name], **kwargs)
            except TypeError as exc:
                if language:
                    raise ValueError(
                        f"this NeMo build does not take a language in transcribe() ({exc}); "
                        "leave 'language' out and the model detects it"
                    ) from exc
                raise

        hyp = _first_hypothesis(out)
        text = (getattr(hyp, "text", None) or str(hyp)).strip()
        tags = LANG_TAG.findall(text)
        return {
            "text": LANG_TAG.sub(" ", text).strip(),
            "language": language or (tags[-1] if tags else None),
            "segments": _segments(hyp) if want_ts else [],
            "duration_s": round(duration_s, 2),
            "att_context_size": self.att_context_size,
            "model_id": self.model_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model_id": self.model_id, "device": self.device,
            "att_context_size": self.att_context_size,
        }


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
        {
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "text": LANG_TAG.sub(" ", str(s["segment"])).strip(),
        }
        for s in stamps
    ]
