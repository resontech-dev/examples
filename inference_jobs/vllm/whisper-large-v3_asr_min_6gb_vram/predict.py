"""
Transcribe audio via the deployed the Whisper large-v3 transcription endpoint endpoint.

Uses the OpenAI transcription API surface:

    POST /v1/audio/transcriptions   (multipart: file + model)

⚠️ Validate on the pinned stack first — Whisper support through Ray Serve
LLM's OpenAI router is the one route in this repo flagged "verify before
listing" (see README.md). If this script gets a 404, the route isn't
exposed by the deployed ray-llm image version.

Run
---
    pip install openai python-dotenv
    python predict.py                          # transcribes sample_data/hello_reson.wav
    python predict.py ./meeting.wav
"""
from __future__ import annotations

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
    """Normalize the platform endpoint to an OpenAI base_url (…/v1)."""
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def _client() -> OpenAI:
    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    return OpenAI(
        base_url=url,
        api_key=key,                             # → Authorization: Bearer (vLLM layer)
        default_headers={"X-API-Key": key},      # → platform serve-proxy layer
        timeout=float(os.getenv("PREDICT_TIMEOUT", "300")),
    )


def _list_models(client: OpenAI) -> list[str]:
    url = str(client.base_url)
    try:
        return [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {url} — deployment not RUNNING yet, or URL wrong: {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(
                f"[predict] {exc.status_code} from {url} — check RESON_INFERENCE_API_KEY "
                "(the job-scoped predict key from submit.py, not the platform rsk_ key)."
            )
        if exc.status_code == 404:
            sys.exit(
                f"[predict] 404 at {url}/models — endpoint is up but has no OpenAI "
                "surface. Was the job submitted with engine: vllm_openai?"
            )
        raise


DEFAULT_AUDIO = HERE / "sample_data" / "hello_reson.wav"


def main() -> None:
    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not audio.is_file():
        sys.exit(f"[predict] audio file not found: {audio}")

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list — replica still loading weights?")
    model = os.getenv("MODEL_ID", models[0])
    print(f"[predict] models: {models} → using {model!r}")
    print(f"[predict] audio:  {audio} ({audio.stat().st_size / 1024:.0f} KB)")
    print("─" * 72)

    try:
        with audio.open("rb") as fh:
            result = client.audio.transcriptions.create(model=model, file=fh)
    except APIStatusError as exc:
        if exc.status_code == 404:
            sys.exit(
                "[predict] 404 on /v1/audio/transcriptions — the deployed router "
                "doesn't expose the transcription route. Validate Whisper support "
                "on the pinned ray-llm image (see README.md)."
            )
        raise
    print(result.text)


if __name__ == "__main__":
    main()
