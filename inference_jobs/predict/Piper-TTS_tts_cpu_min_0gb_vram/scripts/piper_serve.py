"""Piper TTS — text in, WAV out, CPU only.

Request (JSON via ``client.predict_json``):

    {"text": "Добрий день! Ваше замовлення в дорозі.",
     "voice": "uk_UA-ukrainian_tts-medium",      # optional, one of init_args.voices
     "length_scale": 1.0,                        # optional: 2.0 = twice as slow
     "noise_scale": 0.667, "noise_w_scale": 0.8, # optional: variation / style
     "volume": 1.0}                              # optional

A bare text body is also accepted and treated as {"text": <body>}.

Response:

    {"audio_b64": "<base64 16-bit WAV>", "sample_rate": 22050,
     "duration_s": 2.1, "voice": "uk_UA-ukrainian_tts-medium", "elapsed_ms": 180.0}

Voices are ONNX files from HF `rhasspy/piper-voices`, resolved from the
name: uk_UA-ukrainian_tts-medium → uk/uk_UA/ukrainian_tts/medium/<name>.onnx
(+ .onnx.json). Piper embeds espeak-ng for phonemisation — nothing to
apt-install.
"""
from __future__ import annotations

import base64
import io
import json
import time
import wave
from typing import Any

HF_REPO = "rhasspy/piper-voices"


def _voice_path(name: str) -> str:
    """'uk_UA-ukrainian_tts-medium' → 'uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium'."""
    try:
        locale, voice, quality = name.split("-", 2)
    except ValueError as exc:
        raise ValueError(f"voice name must look like 'uk_UA-ukrainian_tts-medium', got {name!r}") from exc
    lang = locale.split("_")[0]
    return f"{lang}/{locale}/{voice}/{quality}/{name}"


class PiperTTS:
    def __init__(
        self,
        voices: list[str] | tuple[str, ...] = ("uk_UA-ukrainian_tts-medium",),
        default_voice: str = "uk_UA-ukrainian_tts-medium",
        use_cuda: bool = False,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        if default_voice not in voices:
            raise ValueError(f"default_voice {default_voice!r} is not in voices {list(voices)}")
        self.default_voice = default_voice
        self.voices: dict[str, Any] = {}
        for name in voices:
            base = _voice_path(name)
            onnx = hf_hub_download(HF_REPO, f"{base}.onnx")
            cfg = hf_hub_download(HF_REPO, f"{base}.onnx.json")
            self.voices[name] = PiperVoice.load(onnx, config_path=cfg, use_cuda=use_cuda)

    def predict(self, data: bytes) -> dict[str, Any]:
        from piper import SynthesisConfig

        t0 = time.perf_counter()
        req = _parse_request(data)
        text = (req.get("text") or "").strip()
        if not text:
            raise ValueError("missing 'text' field")
        name = req.get("voice", self.default_voice)
        voice = self.voices.get(name)
        if voice is None:
            raise ValueError(f"unknown voice {name!r}; loaded: {sorted(self.voices)}")

        syn = SynthesisConfig(
            length_scale=float(req.get("length_scale", 1.0)),
            noise_scale=float(req.get("noise_scale", 0.667)),
            noise_w_scale=float(req.get("noise_w_scale", 0.8)),
            volume=float(req.get("volume", 1.0)),
        )

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            voice.synthesize_wav(text, wav, syn_config=syn)
        wav_bytes = buf.getvalue()
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            sample_rate = wav.getframerate()
            duration_s = wav.getnframes() / float(sample_rate or 1)

        return {
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sample_rate,
            "duration_s": round(duration_s, 2),
            "voice": name,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def warmup(self) -> None:
        self.predict('{"text": "Перевірка."}'.encode("utf-8"))

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "voices": sorted(self.voices), "default_voice": self.default_voice}


def _parse_request(data: bytes) -> dict[str, Any]:
    try:
        req = json.loads(data.decode("utf-8"))
        if isinstance(req, dict):
            return req
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {"text": data.decode("utf-8", errors="replace")}
