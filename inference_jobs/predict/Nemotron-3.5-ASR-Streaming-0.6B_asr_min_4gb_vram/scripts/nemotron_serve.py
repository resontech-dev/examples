"""Nemotron 3.5 ASR Streaming 0.6B: live speech-to-text in sessions, or whole files (NeMo, predictor mode).

A cache-aware streaming FastConformer + RNN-T model (NVIDIA, OpenMDW-1.1):
built to transcribe live audio in chunks of 80 ms to 1.12 s, in 40 language
locales, 19 of them "transcription-ready", Ukrainian (uk-UA) among them.

Live sessions (the platform has no WebSocket route, so one HTTP request per
piece of audio; the encoder cache of each session stays on the GPU between
requests, in NeMo's cache-aware streaming pipeline):

    {"action": "start", "language": "uk-UA"}     # optional; default: init_args.live_language
      -> {"session": "<id>", "frame_ms": 560, "sample_rate": 16000,
          "encoding": "pcm_s16le", "language": "uk-UA", "languages": [...], ...}
    {"action": "append", "session": "<id>", "audio_b64": "<raw PCM16 LE, 16 kHz, mono>"}
      -> {"session": "<id>", "text": "<finalized text so far>",
          "partial": "<words still being decided>", "delta": "<text finalized by this call>",
          "final": false, "language": "uk-UA", "audio_s": 4.2, "elapsed_ms": 31.0}
    {"action": "append", "session": "<id>", "audio_b64": "...", "final": true}
    {"action": "stop", "session": "<id>"}         # the same as append with final: true
      -> the same reply with "final": true; the session is closed

Send one append at a time per session and wait for its reply. Any amount of
audio per append (up to 30 s): the server cuts it into frame_ms frames and
keeps the remainder for the next call. A session idle for session_idle_s is
closed. "start" may carry audio_b64 too.

Two passes (init_args.final_model: parakeet | whisper | whisper-turbo): Nemotron
streams the draft; when it finishes a phrase (~0.8 s of silence), the phrase's
audio goes to the final model in a background worker, and its text replaces the
draft. The protocol stays the same: `text` only ever grows, now with the final
model's text; `partial` holds the drafts still being re-transcribed plus the
words being spoken; the closing call waits for the last phrases. Whole files go
to the final model.

Whole files (request/response):

  - the audio file itself as the body (WAV, FLAC, OGG or MP3);
  - JSON via ``client.predict_json``:

        {"audio_b64": "<base64 of an audio file>",
         "language": "uk-UA",        # optional; default: "auto" (the model detects it)
         "timestamps": false}        # optional

Languages, on both routes: one of the model's prompt keys ("uk-UA", "en-US",
"en"), or an ISO code that maps to the first key it starts ("uk" -> "uk-UA").
An unknown one is a 400 whose message starts "unknown language".

    -> {"text": "Добрий день…", "language": "uk-UA", "segments": [],
        "duration_s": 12.3, "att_context_size": [56, 13],
        "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b", "elapsed_ms": 640.0}

With automatic language detection the model appends a tag such as
``<uk-UA>`` to the text; it is cut off and returned as ``language``.

VALIDATE BEFORE LISTING: written, not booted. VERIFY list in README.md.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field
from math import gcd
from typing import Any

import numpy as np


SAMPLE_RATE = 16_000
LANG_TAG = re.compile(r"\s*<([a-z]{2,3}-[A-Z]{2})>\s*")
LIVE_ACTIONS = ("start", "append", "stop")
MAX_APPEND_S = 30.0
FINAL_WAIT_S = 60.0          # a closing call waits this long for the second pass of its last phrases
KEEP_SILENCE_S = 2.0         # audio kept before speech when nothing has been heard for a while
log = logging.getLogger(__name__)

# Second-pass models: Nemotron streams the draft, one of these writes the final
# text of every finished phrase (and transcribes whole files).
FINAL_MODELS = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",           # NeMo; FLEURS uk WER 6.8 % (card)
    "whisper": "openai/whisper-large-v3",                # transformers
    "whisper-turbo": "openai/whisper-large-v3-turbo",    # transformers; ~2x faster, a little less accurate
}


class NemotronStreamingASR:
    def __init__(
        self,
        model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        att_context_size: list[int] | None = None,
        timestamps: bool = False,
        live: bool = True,
        live_att_context_size: list[int] | None = None,
        live_language: str | None = None,
        max_sessions: int = 32,
        session_idle_s: float = 60.0,
        final_model: str | None = None,
        final_model_id: str | None = None,
    ) -> None:
        import torch

        self.model_id = model_id
        self.timestamps = bool(timestamps)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.att_context_size = list(att_context_size or [56, 13])
        self._lock = threading.Lock()
        self.model = None
        self.languages: list[str] = []   # the language prompts Nemotron takes ([] for a model without them)
        self._TranscribeConfig = None

        # Two passes: the final model writes whole files and the final text of
        # every live phrase; Nemotron only streams, so its file copy is not loaded.
        self.final = make_finalizer(final_model, final_model_id, self.device) if final_model else None
        if self.final is None:
            import nemo.collections.asr as nemo_asr

            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location=self.device)
            self.model.eval()
            # [left, right] context in 80 ms frames; the right one sets the latency:
            # 0 -> 80 ms, 1 -> 160 ms, 3 -> 320 ms, 6 -> 560 ms, 13 -> 1.12 s (most accurate).
            encoder = getattr(self.model, "encoder", None)
            if encoder is not None and hasattr(encoder, "set_default_att_context_size"):
                encoder.set_default_att_context_size(self.att_context_size)
            self.languages = _prompt_keys(self.model)
            if self.languages:
                from nemo.collections.asr.models.rnnt_bpe_models_prompt import RNNTPromptTranscribeConfig

                self._TranscribeConfig = RNNTPromptTranscribeConfig

        # Live sessions run on a copy of Nemotron inside NeMo's streaming
        # pipeline, which sets its own context size and decoding.
        self.live = (
            LiveSessions(
                model_id=model_id,
                att_context_size=list(live_att_context_size or [56, 6]),
                language=live_language,
                max_sessions=int(max_sessions),
                idle_s=float(session_idle_s),
                device=self.device,
                finalizer=self.final,
            )
            if live
            else None
        )
        if self.final is not None and self.live is not None:
            self.languages = self.live.languages   # the same mapping validates file languages

    def predict(self, data: bytes) -> dict[str, Any]:
        req = _parse_json(data)
        if req is not None and req.get("action") is not None:
            if self.live is None:
                raise ValueError("live sessions are off on this job (init_args.live: false)")
            return self.live.handle(req)
        return self._transcribe_file(data, req)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "model_id": self.model_id, "device": self.device,
            "att_context_size": self.att_context_size, "languages": self.languages,
            "final_model": self.final.model_id if self.final else None,
            "live": self.live.info() if self.live else None,
        }

    def _prompt_key(self, language: str | None) -> str | None:
        return resolve_language(language, self.languages)

    # ── whole files ─────────────────────────────────────────────────────────

    def _transcribe_file(self, data: bytes, req: dict[str, Any] | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        if req is not None:
            b64 = req.get("audio_b64") or req.get("data_b64")   # data_b64: client.predict_b64
            if not b64:
                raise ValueError(
                    "a JSON body needs 'audio_b64' (the base64 of an audio file) or 'action' (a live session)"
                )
            audio_bytes, opts = base64.b64decode(b64), req
        else:
            audio_bytes, opts = data, {}
        samples = _decode_16k_mono(audio_bytes)
        duration_s = len(samples) / SAMPLE_RATE
        if duration_s < 0.1:
            raise ValueError("the audio is empty or shorter than 0.1 s")
        want_ts = bool(opts.get("timestamps", self.timestamps))
        if self.languages or self.final is None:
            # "uk" -> "uk-UA"; None -> "auto"; any language on a model without prompts is an error
            language = self._prompt_key(opts.get("language") or None)
        else:
            language = str(opts.get("language") or "").strip() or None   # final model, no live copy to map with

        if self.final is not None:
            text, segments = self.final.transcribe(samples, _iso(language), want_ts)
            return {
                "text": text,
                "language": language if language not in (None, "auto") else None,
                "segments": segments,
                "duration_s": round(duration_s, 2),
                "att_context_size": None,
                "model_id": self.final.model_id,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        with self._lock, tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            _write_wav(tmp.name, samples)
            if self._TranscribeConfig is None:
                out = self.model.transcribe([tmp.name], batch_size=1, timestamps=want_ts, verbose=False)
            else:
                # NeMo 3.0.0's default Lhotse dataset takes the language from the
                # manifest's `lang` field, which transcribe() never writes, and
                # ignores target_lang: half the files then failed with "Unknown
                # prompt key: 'None'", the other half ran as auto. Without Lhotse
                # the batch carries no prompt index, and _transcribe_forward
                # looks target_lang up in the model's prompt_dictionary.
                trcfg = self._TranscribeConfig(
                    batch_size=1,
                    num_workers=0,                 # None would start a DataLoader worker per file
                    verbose=False,
                    timestamps=want_ts,
                    return_hypotheses=want_ts,     # transcribe() sets it from `timestamps` only without override_config
                    use_lhotse=False,
                    target_lang=language,
                )
                # `timestamps` again as an argument: transcribe() switches the
                # decoder on it, not on override_config.
                out = self.model.transcribe([tmp.name], timestamps=want_ts, override_config=trcfg)

        hyp = _first_hypothesis(out)
        text = _hyp_text(hyp)
        tags = LANG_TAG.findall(text)
        return {
            "text": _strip_tags(text).strip(),
            "language": language if language not in (None, "auto") else (tags[-1] if tags else None),
            "segments": _segments(hyp) if want_ts else [],
            "duration_s": round(duration_s, 2),
            "att_context_size": self.att_context_size,
            "model_id": self.model_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


# ── live sessions ───────────────────────────────────────────────────────────


@dataclass
class _Phrase:
    """A phrase Nemotron finished: its draft, and the second pass that will replace it."""

    draft: str
    future: Any
    text: str | None = None


@dataclass
class _Session:
    stream_id: int
    language: str | None
    options: Any
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    started: bool = False
    text: str = ""                 # Nemotron's finalized text (the draft, with two passes)
    partial: str = ""
    detected: str | None = None
    audio_s: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    reported: str = ""             # `text` as the previous reply gave it, for `delta`
    phrase_audio: list = field(default_factory=list)   # two passes: audio of the phrase being spoken
    phrase_samples: int = 0
    phrases: list = field(default_factory=list)        # two passes: finished phrases, in order


class LiveSessions:
    """Live transcription sessions on NeMo's cache-aware streaming pipeline, one state per session."""

    def __init__(
        self,
        model_id: str,
        att_context_size: list[int],
        language: str | None,
        max_sessions: int,
        idle_s: float,
        device: str,
        finalizer: Any = None,
    ) -> None:
        import torch
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
        from omegaconf import OmegaConf

        self._torch, self._Frame, self._Options = torch, Frame, ASRRequestOptions
        # float32, not the reference config's bfloat16: NeMo 3.0.0's wrapper casts
        # the whole model to this dtype, and in bfloat16 the one-hot language
        # prompt is rounded away inside the prompt projection (NeMo's own note in
        # #16047). A 0.6B model in fp32 still takes tens of ms per frame.
        cfg = _live_config(model_id, att_context_size, max_sessions, device, "float32")
        self.pipeline = PipelineBuilder.build_pipeline(OmegaConf.create(cfg))
        self.pipeline.open_session()
        self.sep = self.pipeline.get_sep()
        self.prompt_patch = _patch_prompt_injection(self.pipeline)

        self.att_context_size = att_context_size
        self.frame_samples = int(round(self.pipeline.chunk_size_in_secs * SAMPLE_RATE))
        self.frame_ms = int(round(self.pipeline.chunk_size_in_secs * 1000))
        self.languages = _pipeline_prompt_keys(self.pipeline)   # [] when the model takes no language prompt
        # live_language through the same mapping as every request: None -> "auto", "uk" -> "uk-UA".
        self.default_language = resolve_language(language, self.languages)

        self.max_sessions = max_sessions
        self.idle_s = idle_s
        self._sessions: dict[str, _Session] = {}
        self._next_stream_id = 0
        self._lock = threading.Lock()
        # Two passes: one worker re-transcribes finished phrases in the order
        # they end, so a slow second pass never holds up the streaming.
        self.final = finalizer
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-final") if finalizer else None

    def info(self) -> dict[str, Any]:
        return {
            "sessions": len(self._sessions), "max_sessions": self.max_sessions,
            "frame_ms": self.frame_ms, "att_context_size": self.att_context_size,
            "default_language": self.default_language, "languages": self.languages,
            "final_model": self.final.model_id if self.final else None,
            "prompt_patch": self.prompt_patch,
        }

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        action = req.get("action")
        if action not in LIVE_ACTIONS:
            raise ValueError(f"action must be one of {', '.join(LIVE_ACTIONS)}, got {action!r}")
        pcm = _decode_pcm16(req.get("audio_b64"))
        if len(pcm) > MAX_APPEND_S * SAMPLE_RATE:
            raise ValueError(f"at most {MAX_APPEND_S:.0f} s of audio per call; send smaller pieces")
        final = action == "stop" or bool(req.get("final"))

        with self._lock:
            self._close_idle()
            if action == "start":
                session_id, session = self._open(req.get("language"))
            else:
                session_id = str(req.get("session") or "")
                session = self._sessions.get(session_id)
                if session is None:
                    raise ValueError(
                        "unknown session: it was closed (final), idle for more than "
                        f"{self.idle_s:.0f} s, or the job restarted; start a new one"
                    )
            if not (action == "start" and pcm.size == 0 and not final):
                self._feed(session_id, session, pcm, final)
            closing_two_pass = final and self.final is not None
            if not closing_two_pass:
                reply = self._reply(session_id, session, final)
        if closing_two_pass:
            # Outside the lock, so other sessions keep streaming: the last
            # phrases' second pass finishes before the closing reply.
            futures_wait([p.future for p in session.phrases], timeout=FINAL_WAIT_S)
            reply = self._reply(session_id, session, final)

        if action == "start":
            reply.update(frame_ms=self.frame_ms, sample_rate=SAMPLE_RATE, encoding="pcm_s16le",
                         att_context_size=self.att_context_size, languages=self.languages,
                         final_model=self.final.model_id if self.final else None)
        reply["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return reply

    def _open(self, language: str | None) -> tuple[str, _Session]:
        language = resolve_language(language or self.default_language, self.languages)
        options = self._Options(language_code=language) if self.languages else self._Options()
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"all {self.max_sessions} live sessions are busy; retry in a moment")
        session = _Session(stream_id=self._next_stream_id, language=language, options=options)
        self._next_stream_id += 1
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = session
        return session_id, session

    def _feed(self, session_id: str, session: _Session, pcm: np.ndarray, final: bool) -> None:
        session.last_seen = time.monotonic()
        session.audio_s += len(pcm) / SAMPLE_RATE
        samples = np.concatenate([session.pending, pcm]) if session.pending.size else pcm
        n = self.frame_samples
        full = len(samples) // n
        chunks = [(samples[i * n:(i + 1) * n], n) for i in range(full)]
        rest = samples[full * n:]
        if final:
            if rest.size or not chunks:
                # The last frame is zero-padded to the frame size, as NeMo's own
                # streamer does; with no audio left it is one frame of silence.
                last = np.zeros(n, dtype=np.float32)
                last[:rest.size] = rest
                chunks.append((last, rest.size or n))
            session.pending = np.zeros(0, dtype=np.float32)
        else:
            session.pending = rest

        try:
            for i, (chunk, valid) in enumerate(chunks):
                self._step(session, chunk, valid, is_last=final and i == len(chunks) - 1)
        except Exception:
            self._drop(session_id, session)
            raise
        if final:
            del self._sessions[session_id]
            session.partial = ""

    def _step(self, session: _Session, chunk: np.ndarray, valid: int, is_last: bool) -> None:
        frame = self._Frame(
            samples=self._torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)),
            stream_id=session.stream_id,
            is_first=not session.started,
            is_last=is_last,
            length=valid,
            options=None if session.started else session.options,
        )
        if self.final is not None and valid:
            session.phrase_audio.append(np.array(chunk[:valid], dtype=np.float32))   # a copy: chunk is a view
            session.phrase_samples += valid
        with self._torch.inference_mode():
            out = self.pipeline.transcribe_step([frame])[0]
        session.started = True
        piece = out.final_transcript
        if not session.text:
            piece = piece.lstrip(self.sep)
        session.text += piece
        session.partial = out.partial_transcript
        tags = LANG_TAG.findall(piece + " " + out.partial_transcript)
        if tags:
            session.detected = tags[-1]
        if self.final is not None:
            self._phrase_boundary(session, _strip_tags(piece).strip())

    def _phrase_boundary(self, session: _Session, draft: str) -> None:
        """Two passes: a phrase Nemotron just finished goes to the final model with its audio."""
        if draft:
            # Nemotron finalizes after ~0.8 s of silence (or at the session's
            # end), so the cut falls in a pause, not inside a word.
            audio = np.concatenate(session.phrase_audio)
            session.phrase_audio, session.phrase_samples = [], 0
            # A session on "auto" lets the final model detect each phrase's language
            # itself, so a speaker who switches languages gets each phrase right.
            language = None if session.language in (None, "auto") else session.language
            future = self._pool.submit(self._final_text, audio, _iso(language), draft)
            session.phrases.append(_Phrase(draft=draft, future=future))
        elif not session.partial.strip() and session.phrase_samples > 2 * KEEP_SILENCE_S * SAMPLE_RATE:
            # Nothing heard yet: keep only the last seconds before the next phrase.
            audio = np.concatenate(session.phrase_audio)[-int(KEEP_SILENCE_S * SAMPLE_RATE):]
            session.phrase_audio, session.phrase_samples = [audio], len(audio)

    def _final_text(self, audio: np.ndarray, language: str | None, draft: str) -> str:
        try:
            text, _ = self.final.transcribe(audio, language, False)
        except Exception:   # the draft stands in; the session goes on
            log.exception("second pass failed; keeping the streaming draft")
            return draft
        return text or draft

    def _reply(self, session_id: str, session: _Session, final: bool) -> dict[str, Any]:
        if self.final is None:
            text = _strip_tags(session.text).strip()
            partial = _strip_tags(session.partial).strip()
        else:
            # The final model's text for the phrases done in order; the drafts
            # of the rest, then the words being spoken, stay in `partial`.
            confirmed, drafts = [], []
            for phrase in session.phrases:
                if phrase.text is None and not drafts and phrase.future.done():
                    phrase.text = phrase.future.result()
                if phrase.text is None and final:
                    phrase.text = phrase.draft   # the second pass ran out of FINAL_WAIT_S
                if phrase.text is not None and not drafts:
                    confirmed.append(phrase.text)
                else:
                    drafts.append(phrase.draft)
            text = " ".join(t for t in confirmed if t)
            partial = " ".join(t for t in (*drafts, _strip_tags(session.partial).strip()) if t)
        delta = text[len(session.reported):] if text.startswith(session.reported) else text
        session.reported = text
        language = session.language
        if language in (None, "auto"):
            language = session.detected or language
        return {
            "session": session_id,
            "text": text,
            "partial": partial,
            "delta": delta,
            "final": final,
            "language": language,
            "audio_s": round(session.audio_s, 2),
        }

    def _drop(self, session_id: str, session: _Session) -> None:
        """Close a session that will not finish: one silent last frame frees its pipeline slot."""
        self._sessions.pop(session_id, None)
        if not session.started:
            return
        try:
            silence = np.zeros(self.frame_samples, dtype=np.float32)
            frame = self._Frame(
                samples=self._torch.from_numpy(silence), stream_id=session.stream_id,
                is_first=False, is_last=True, length=self.frame_samples,
            )
            with self._torch.inference_mode():
                self.pipeline.transcribe_step([frame])
        except Exception:   # the slot is lost at worst; the reply that failed already carries the error
            pass

    def _close_idle(self) -> None:
        now = time.monotonic()
        for session_id, session in list(self._sessions.items()):
            if now - session.last_seen > self.idle_s:
                self._drop(session_id, session)


# ── second pass ─────────────────────────────────────────────────────────────


def make_finalizer(name: str, model_id: str | None, device: str) -> Any:
    """The second-pass model named by init_args.final_model (model_id overrides its checkpoint)."""
    key = str(name).strip().lower()
    if key not in FINAL_MODELS:
        raise ValueError(f"final_model must be one of {', '.join(FINAL_MODELS)} or null, got {name!r}")
    model_id = model_id or FINAL_MODELS[key]
    if key == "parakeet":
        return NemoFinalizer(key, model_id, device)
    return WhisperFinalizer(key, model_id, device)


class NemoFinalizer:
    """A NeMo checkpoint (Parakeet v3 by default) as the second pass. It detects the language itself."""

    def __init__(self, name: str, model_id: str, device: str) -> None:
        import nemo.collections.asr as nemo_asr

        self.name, self.model_id = name, model_id
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location=device)
        self.model.eval()
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, language: str | None, timestamps: bool) -> tuple[str, list]:
        with self._lock, tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            _write_wav(tmp.name, samples)
            out = self.model.transcribe([tmp.name], batch_size=1, timestamps=timestamps, verbose=False)
        hyp = _first_hypothesis(out)
        return _strip_tags(_hyp_text(hyp)).strip(), (_segments(hyp) if timestamps else [])


class WhisperFinalizer:
    """Whisper (transformers) as the second pass; `language` is an ISO code, None to detect it."""

    def __init__(self, name: str, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.name, self.model_id, self.device = name, model_id, device
        self._torch = torch
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Loaded in fp32 and cast: the dtype keyword was renamed between transformers 4 and 5.
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device, dtype=self.dtype).eval()
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, language: str | None, timestamps: bool) -> tuple[str, list]:
        long_form = len(samples) > 30 * SAMPLE_RATE
        if long_form:   # transformers' sequential long-form decoding over 30 s windows
            inputs = self.processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", truncation=False,
                                    padding="longest", return_attention_mask=True)
        else:
            inputs = self.processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                    return_attention_mask=True)
        inputs = inputs.to(self.device, self.dtype)
        kwargs: dict[str, Any] = {"task": "transcribe", "return_timestamps": bool(timestamps or long_form)}
        if language:
            kwargs["language"] = language
        with self._lock, self._torch.inference_mode():
            ids = self.model.generate(**inputs, **kwargs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        segments: list = []
        if timestamps:
            try:
                decoded = self.processor.batch_decode(ids, skip_special_tokens=True, output_offsets=True)[0]
                segments = [
                    {"start": round(float(o["timestamp"][0]), 2),
                     "end": round(float(o["timestamp"][1] if o["timestamp"][1] is not None else o["timestamp"][0]), 2),
                     "text": o["text"].strip()}
                    for o in decoded.get("offsets", [])
                ]
            except Exception:   # timestamps are a bonus; the text stands without them
                log.warning("whisper offsets unavailable", exc_info=True)
        return text, segments


def _iso(language: str | None) -> str | None:
    """A prompt key or locale as the ISO code Whisper takes: "uk-UA" -> "uk"; "auto" / None -> None."""
    if not language or language == "auto":
        return None
    return language.split("-")[0].lower()


def _live_config(model_id: str, att_context_size: list[int], max_sessions: int, device: str, dtype: str) -> dict:
    """NeMo 3.0.0's examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml, trimmed to the keys the
    pipeline reads, with ITN and translation off."""
    return {
        "asr": {
            "model_name": model_id,
            "device": device,
            "device_id": 0,
            "compute_dtype": dtype,
            "use_amp": False,
            "decoding": {
                "strategy": "greedy_batch",
                "preserve_alignments": False,
                "fused_batch_size": -1,
                "greedy": {
                    "use_cuda_graph_decoder": False,   # the reference config: "issues with decoding"
                    "enable_per_stream_biasing": True,
                    "preserve_frame_confidence": False,
                    "max_symbols": 10,
                    "ngram_lm_model": None,
                    "ngram_lm_alpha": 0.0,
                    "boosting_tree": {
                        "model_path": None, "key_phrases_file": None, "key_phrases_list": None,
                        "key_phrase_items_list": None, "source_lang": "en",
                    },
                    "boosting_tree_alpha": 0.0,
                },
            },
        },
        "itn": {
            "input_case": "lower_cased", "whitelist": None, "overwrite_cache": False,
            "max_number_of_permutations_per_split": 729, "left_padding_size": 4, "batch_size": 32, "n_jobs": 16,
        },
        "confidence": {
            "exclude_blank": True,
            "aggregation": "mean",
            "method_cfg": {"name": "entropy", "entropy_type": "tsallis", "alpha": 0.5, "entropy_norm": "exp"},
        },
        "endpointing": {"stop_history_eou": 800, "residue_tokens_at_end": 2},
        "streaming": {
            "sample_rate": SAMPLE_RATE,
            "batch_size": min(16, max_sessions),
            "word_boundary_tolerance": 4,
            "att_context_size": att_context_size,
            "use_cache": True,
            "use_feat_cache": True,
            "chunk_size_in_secs": None,       # 1 + right context frames of 80 ms
            "request_type": "frame",
            "num_slots": max_sessions,
        },
        "matmul_precision": "high",
        "log_level": 30,
        "pipeline_type": "cache_aware",
        "asr_decoding_type": "rnnt",
        "enable_itn": False,
        "enable_nmt": False,
        "asr_output_granularity": "segment",
        "cache_dir": None,
        "lang": None,
        "return_tail_result": False,
    }


def _patch_prompt_injection(pipeline: Any) -> bool:
    """Apply the per-stream language prompts NeMo 3.0.0's streaming wrapper drops.

    The pipeline builds a one-hot language vector per stream and passes it to
    CacheAwareRNNTInferenceWrapper.stream_step(prompt_vectors=...), but 3.0.0's
    execute_step never uses it: the decoder of a prompt model then gets encoder
    output without the language projection it was trained on, and emits no
    tokens (live sessions stayed empty while whole files worked). NeMo fixed it
    after 3.0.0 (#16047, 2026-08-21) by projecting the encoder output right
    after the encoder step; this does the same, on a NeMo without that fix.
    Returns True when the patch is installed.
    """
    import torch

    wrapper = getattr(pipeline, "asr_model", None)
    model = getattr(wrapper, "asr_model", None)
    if not getattr(pipeline, "prompt_enabled", False) or not getattr(model, "concat", False):
        return False
    if hasattr(wrapper, "_apply_prompt_vectors"):   # a NeMo with the upstream fix
        return False
    kernel = model.prompt_kernel
    first = next((m for m in kernel.modules() if isinstance(m, torch.nn.Linear)), None)
    enc_hidden = model.cfg.get("model_defaults", {}).get("enc_hidden")
    if first is not None and enc_hidden is not None and first.in_features != int(model.num_prompts) + int(enc_hidden):
        # Fail at boot rather than mis-condition every session if the scheme changed.
        raise RuntimeError(
            f"unexpected prompt_kernel input size {first.in_features}, expected num_prompts + enc_hidden = "
            f"{int(model.num_prompts) + int(enc_hidden)}; the prompt patch no longer matches this NeMo"
        )

    current: dict[str, Any] = {"prompts": None}
    encoder_step = model.encoder.cache_aware_stream_step
    stream_step = wrapper.stream_step

    def cache_aware_stream_step(*args: Any, **kwargs: Any) -> Any:
        out = encoder_step(*args, **kwargs)
        if current["prompts"] is None:
            return out
        return (_inject_prompts(kernel, out[0], current["prompts"]), *out[1:])

    def stream_step_with_prompts(*args: Any, prompt_vectors: Any = None, **kwargs: Any) -> Any:
        current["prompts"] = prompt_vectors
        try:
            return stream_step(*args, prompt_vectors=prompt_vectors, **kwargs)
        finally:
            current["prompts"] = None

    model.encoder.cache_aware_stream_step = cache_aware_stream_step
    wrapper.stream_step = stream_step_with_prompts
    return True


def _inject_prompts(kernel: Any, encoded: Any, prompts: Any) -> Any:
    """[B, D, T] encoder output + [B, num_prompts] one-hot prompts -> the projected [B, D, T].

    The batched form of PromptStreamingMixin._apply_prompt_to_encoded (concat
    [encoded, prompt] per frame, then prompt_kernel), as NeMo main does it, in
    float32 with autocast off so the one-hot signal is not rounded away.
    """
    import torch

    encoded = encoded.transpose(1, 2)   # (B, D, T) -> (B, T, D)
    out_dtype = encoded.dtype
    prompt = prompts.to(dtype=torch.float32, device=encoded.device).unsqueeze(1).expand(-1, encoded.shape[1], -1)
    weight_dtype = next(kernel.parameters()).dtype
    with torch.autocast(device_type=encoded.device.type, enabled=False):
        projected = kernel(torch.cat([encoded.float(), prompt], dim=-1).to(weight_dtype))
    return projected.to(out_dtype).transpose(1, 2)   # (B, T, D) -> (B, D, T)


def _pipeline_prompt_keys(pipeline: Any) -> list[str]:
    """The live pipeline's language-prompt keys, with "auto" added the same way as on the file model."""
    if not getattr(pipeline, "prompt_enabled", False):
        return []
    model = getattr(getattr(pipeline, "asr_model", None), "asr_model", None)
    if model is not None and _prompt_keys(model) and hasattr(pipeline, "_load_prompt_config"):
        pipeline._prompt_config = pipeline._load_prompt_config()   # re-read: picks up an added "auto"
    prompt_dict = (getattr(pipeline, "_prompt_config", None) or {}).get("prompt_dict") or {}
    return [str(k) for k in prompt_dict.keys()]


# ── languages (one mapping for files and live sessions) ─────────────────────

AUTO_PROMPT_INDEX = 101


def _prompt_keys(model: Any) -> list[str]:
    """The model's language-prompt keys in its own order, "auto" added when missing; [] without prompts."""
    cfg = getattr(model, "cfg", None)
    defaults = cfg.get("model_defaults") if cfg is not None and hasattr(cfg, "get") else None
    prompt_dict = defaults.get("prompt_dictionary") if defaults is not None else None
    if not prompt_dict:
        return []
    if "auto" not in prompt_dict and int(defaults.get("num_prompts", 128)) > AUTO_PROMPT_INDEX:
        # NeMo's prompt-index dataset uses index 101 for "auto" when the dictionary
        # has no such key (LhotseSpeechToTextBpeDatasetWithPromptIndex:
        # prompt_dict.get('auto', 101)); the files that did not fail on the Lhotse
        # route ran with it. VERIFY the auto text on first boot.
        from omegaconf import open_dict

        with open_dict(prompt_dict):
            prompt_dict["auto"] = AUTO_PROMPT_INDEX
    return [str(k) for k in prompt_dict.keys()]


def resolve_language(language: str | None, keys: list[str]) -> str | None:
    """A request's language as one of the model's prompt keys.

    None / "" -> "auto" (or "en-US" without it); an exact key ("uk-UA", "en") as
    is; an ISO code ("uk") -> the first key that starts with "uk-".
    """
    language = (language or "").strip()
    if not keys:
        if language:
            raise ValueError(f"unknown language {language!r}; this model takes no language setting")
        return None
    if not language:
        return "auto" if "auto" in keys else "en-US"
    if language in keys:
        return language
    for key in keys:
        if key.startswith(language + "-"):
            return key
    raise ValueError(f"unknown language {language!r}; this model takes: {', '.join(keys)}")


def _decode_pcm16(b64: str | None) -> np.ndarray:
    if not b64:
        return np.zeros(0, dtype=np.float32)
    try:
        raw = base64.b64decode(b64)
    except ValueError as exc:
        raise ValueError("audio_b64 is not base64") from exc
    if len(raw) % 2:
        raise ValueError("audio_b64 must be raw PCM16 little-endian (an even number of bytes), 16 kHz mono")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


# ── shared helpers ──────────────────────────────────────────────────────────


def _parse_json(data: bytes) -> dict[str, Any] | None:
    """The JSON envelope, or None when the body is the audio file itself."""
    if data[:1] != b"{":
        return None
    try:
        req = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return req if isinstance(req, dict) else None


def _strip_tags(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", LANG_TAG.sub(" ", text))


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


def _hyp_text(hyp: Any) -> str:
    """A hypothesis' text; an empty one stays empty (str(hyp) would print the object)."""
    text = hyp.text if hasattr(hyp, "text") else hyp
    return str(text or "").strip()


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
            "text": _strip_tags(str(s["segment"])).strip(),
        }
        for s in stamps
    ]
