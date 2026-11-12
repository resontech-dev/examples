"""
Transcribe audio via the deployed Qwen3-ASR-1.7B endpoint.

Two routes, same engine:

    --chat (default)   POST /v1/chat/completions with an `audio_url` content
                       part (base64 data URL). The primary route in the vLLM
                       recipe and the model card. The reply carries a language
                       tag followed by the transcript; this script splits it.
    --transcribe       POST /v1/audio/transcriptions (OpenAI transcription
                       API, multipart file + model). Returns plain text.
                       VALIDATE on the pinned stack: a 404 here means the
                       Ray Serve LLM router does not expose the route (README).

Qwen3-ASR identifies the language itself; `--language` only forwards a
hint on the transcription route (the chat route has no language field,
the model prints the detected one). Ukrainian is not a supported language
of this model: use the Whisper jobs for Ukrainian audio.

Run
---
    pip install openai python-dotenv
    python predict.py                                   # sample_data/hello_reson.wav, chat route
    python predict.py ./meeting.wav
    python predict.py ./call.mp3 --transcribe           # OpenAI transcription API
    python predict.py ./call.mp3 --transcribe --language pl
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _openai_base_url(raw: str) -> str:
    """Normalize the platform endpoint to an OpenAI base_url (.../v1)."""
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def _client() -> OpenAI:
    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    return OpenAI(
        base_url=url,
        api_key=key,                             # Authorization: Bearer (vLLM layer)
        default_headers={"X-API-Key": key},      # platform serve-proxy layer
        timeout=float(os.getenv("PREDICT_TIMEOUT", "300")),
    )


def _list_models(client: OpenAI) -> list[str]:
    url = str(client.base_url)
    try:
        return [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {url}: deployment not RUNNING yet, or URL wrong: {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(
                f"[predict] {exc.status_code} from {url}: check RESON_INFERENCE_API_KEY "
                "(the job-scoped predict key from submit.py, not the platform rsk_ key)."
            )
        if exc.status_code == 404:
            sys.exit(
                f"[predict] 404 at {url}/models: endpoint is up but has no OpenAI "
                "surface. Was the job submitted with engine: vllm_openai?"
            )
        raise


DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"


def _audio_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{b64}"}}


def _split_asr_output(content: str) -> tuple[str, str]:
    """Lenient version of qwen_asr.parse_asr_output: language tag, then text.

    The exact tag format is defined by the model's chat template; VERIFY on
    the first real reply and tighten if needed.
    """
    head, _, tail = content.strip().partition("\n")
    if tail and len(head) <= 40:
        lang = head.split(":")[-1].strip() if ":" in head else head.strip()
        return lang, tail.strip()
    return "", content.strip()


def run_chat(client: OpenAI, model: str, audio: Path) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [_audio_part(audio)]}],
        max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    lang, text = _split_asr_output(content)
    if lang:
        print(f"[predict] detected language: {lang}")
    print(text)
    if resp.usage:
        print("-" * 72)
        print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion")


def run_transcribe(client: OpenAI, model: str, audio: Path, language: str | None) -> None:
    kwargs = {"model": model}
    if language:
        kwargs["language"] = language
    try:
        with audio.open("rb") as fh:
            result = client.audio.transcriptions.create(file=fh, **kwargs)
    except APIStatusError as exc:
        if exc.status_code == 404:
            sys.exit(
                "[predict] 404 on /v1/audio/transcriptions: the deployed router "
                "does not expose the transcription route on the pinned image. "
                "Use the chat route (drop --transcribe), see README.md."
            )
        raise
    print(result.text)


def main() -> None:
    argv = sys.argv[1:]
    transcribe = "--transcribe" in argv
    language = None
    if "--language" in argv:
        i = argv.index("--language")
        try:
            language = argv[i + 1]
        except IndexError:
            sys.exit("[predict] --language needs a code (en / it / pl / de ...)")
        argv = argv[:i] + argv[i + 2:]
    positional = [a for a in argv if not a.startswith("--")]
    audio = Path(positional[0]) if positional else DEFAULT_AUDIO
    if not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list: replica still loading weights?")
    model = os.getenv("MODEL_ID", "asr" if "asr" in models else models[0])
    print(f"[predict] models: {models} -> using {model!r}")
    print(f"[predict] audio:  {audio} ({audio.stat().st_size / 1024:.0f} KB), "
          f"route={'transcriptions' if transcribe else 'chat'}"
          + (f", language={language}" if language else ", language=auto"))
    print("-" * 72)

    if transcribe:
        run_transcribe(client, model, audio, language)
    else:
        if language:
            print("[predict] note: the chat route has no language field; the model detects it")
        run_chat(client, model, audio)


if __name__ == "__main__":
    main()
