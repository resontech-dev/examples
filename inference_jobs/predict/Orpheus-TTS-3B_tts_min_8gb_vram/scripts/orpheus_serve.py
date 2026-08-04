"""Orpheus-TTS 3B — text in, 24 kHz speech out.

Request (JSON via ``client.predict_json``):

    {"text": "Hello there!", "voice": "tara",        # optional
     "temperature": 0.6, "top_p": 0.95,              # optional
     "max_new_tokens": 2048}                         # optional

A bare text body is also accepted and treated as {"text": <body>}.

Response:

    {"audio_b64": "<base64 WAV>", "sample_rate": 24000,
     "duration_s": 3.2, "voice": "tara", "elapsed_ms": 4100.0}

How it works: Orpheus is a Llama-3.2-3B fine-tune that emits SNAC audio
codes as special tokens. We generate those tokens, de-interleave them
into the codec's 3 layers, and decode with the SNAC vocoder.

⚠️ VALIDATE BEFORE SELLING: the special-token framing below (start/end
ids, 7-token frames, 128266 offset) follows the canopylabs reference
implementation for orpheus-3b-0.1. It is model-revision-specific — smoke
test on a GPU and listen to the output before listing this template.
Emotion tags like <laugh>, <sigh>, <gasp> work inline in the text.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any

import torch


# Orpheus special-token layout (reference implementation constants).
_START_OF_HUMAN = 128259
_END_OF_TEXT = 128009
_END_OF_HUMAN = 128260
_START_OF_AUDIO = 128257
_END_OF_AUDIO = 128258
_AUDIO_TOKEN_OFFSET = 128266
_CODES_PER_FRAME = 7                 # 1× layer1 + 2× layer2 + 4× layer3
_SAMPLE_RATE = 24000

VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")


class OrpheusTTS:
    def __init__(
        self,
        model_id: str = "canopylabs/orpheus-3b-0.1-ft",
        snac_id: str = "hubertsiuzdak/snac_24khz",
        default_voice: str = "tara",
        max_new_tokens: int = 2048,
        dtype: str = "auto",
    ) -> None:
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.default_voice = default_voice
        self.max_new_tokens = int(max_new_tokens)
        self.device = _pick_device()
        torch_dtype = (
            torch.float16 if (dtype == "auto" and self.device.type == "cuda")
            else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        # SNAC decodes on CPU fine (tiny); keep it off the GPU budget.
        self.snac = SNAC.from_pretrained(snac_id).eval()

    @torch.inference_mode()
    def predict(self, data: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        req = _parse_request(data)
        text = (req.get("text") or "").strip()
        if not text:
            raise ValueError("missing 'text' field")
        voice = req.get("voice", self.default_voice)
        if voice not in VOICES:
            raise ValueError(f"unknown voice {voice!r}; pick one of {VOICES}")

        # 1) Prompt: <SOH> "<voice>: <text>" <EOT> <EOH>
        prompt_ids = self.tokenizer(f"{voice}: {text}", return_tensors="pt").input_ids[0]
        input_ids = torch.cat([
            torch.tensor([_START_OF_HUMAN]),
            prompt_ids,
            torch.tensor([_END_OF_TEXT, _END_OF_HUMAN]),
        ]).unsqueeze(0).to(self.device)

        # 2) Generate SNAC-code tokens.
        out = self.model.generate(
            input_ids,
            max_new_tokens=int(req.get("max_new_tokens", self.max_new_tokens)),
            do_sample=True,
            temperature=float(req.get("temperature", 0.6)),
            top_p=float(req.get("top_p", 0.95)),
            repetition_penalty=1.1,       # <1.1 causes audible looping
            eos_token_id=_END_OF_AUDIO,
        )[0]

        # 3) Crop to the audio span: after the LAST start-of-audio marker,
        #    strip end-of-audio, truncate to whole 7-token frames.
        tokens = out.tolist()
        if _START_OF_AUDIO in tokens:
            tokens = tokens[len(tokens) - tokens[::-1].index(_START_OF_AUDIO):]
        tokens = [t for t in tokens if t not in (_END_OF_AUDIO,)]
        tokens = tokens[: len(tokens) - (len(tokens) % _CODES_PER_FRAME)]
        if not tokens:
            raise ValueError("model produced no audio tokens — try different text/voice")

        audio = self._decode(tokens)

        # 4) WAV-encode.
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, _SAMPLE_RATE, format="WAV", subtype="PCM_16")

        return {
            "audio_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "sample_rate": _SAMPLE_RATE,
            "duration_s": round(len(audio) / _SAMPLE_RATE, 2),
            "voice": voice,
            "model_id": self.model_id,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def _decode(self, tokens: list[int]):
        """De-interleave 7-token frames into SNAC's 3 layers and decode."""
        codes = [t - _AUDIO_TOKEN_OFFSET for t in tokens]
        l1, l2, l3 = [], [], []
        for i in range(0, len(codes), _CODES_PER_FRAME):
            f = codes[i:i + _CODES_PER_FRAME]
            l1.append(f[0])
            l2.append(f[1] - 1 * 4096)
            l3.append(f[2] - 2 * 4096)
            l3.append(f[3] - 3 * 4096)
            l2.append(f[4] - 4 * 4096)
            l3.append(f[5] - 5 * 4096)
            l3.append(f[6] - 6 * 4096)
        layers = [
            torch.tensor(l1).unsqueeze(0),
            torch.tensor(l2).unsqueeze(0),
            torch.tensor(l3).unsqueeze(0),
        ]
        if any((layer < 0).any() or (layer > 4095).any() for layer in layers):
            raise ValueError("audio codes out of range — generation degenerated, retry")
        audio = self.snac.decode(layers)          # [1, 1, samples]
        return audio.squeeze().cpu().numpy()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model_id": self.model_id,
            "voices": list(VOICES), "device": str(self.device),
        }


def _parse_request(data: bytes) -> dict[str, Any]:
    try:
        req = json.loads(data.decode("utf-8"))
        if isinstance(req, dict):
            return req
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {"text": data.decode("utf-8", errors="replace")}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
