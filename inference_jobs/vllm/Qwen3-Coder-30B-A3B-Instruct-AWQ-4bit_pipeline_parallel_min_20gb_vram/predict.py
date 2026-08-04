"""
Talk to the deployed vLLM OpenAI-compatible endpoint.

Unlike the predictor examples (which POST raw bytes to /predict via
``InferenceClient``), a ``vllm_openai`` deployment speaks the OpenAI
protocol — so this script uses the official ``openai`` client, exactly
the way any customer tool (Aider, Cline, Continue, langchain) would:

    POST <endpoint>/v1/chat/completions
    GET  <endpoint>/v1/models

Auth crosses two layers and this script satisfies both:
  * platform edge (serve-proxy nginx) checks ``X-API-Key``  → default_headers
  * vLLM engine may check ``Authorization: Bearer``         → api_key=

Run
---
    pip install openai python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY
    # (printed by submit.py after deploy)

    python predict.py                          # default coding prompt, streamed
    python predict.py "Write a bash one-liner that finds duplicate files"
    python predict.py --no-stream "Explain GIL in two sentences"
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

DEFAULT_PROMPT = "Write a Python function that merges overlapping intervals, with a doctest."


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _openai_base_url(raw: str) -> str:
    """
    Normalize the platform endpoint to an OpenAI base_url.

    submit.py may print the endpoint with a predictor-style ``/predict``
    suffix and/or a trailing slash. The OpenAI surface lives at
    ``<endpoint>/v1``, so: strip both, append ``/v1``.
    """
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stream = "--no-stream" not in sys.argv
    prompt = args[0] if args else DEFAULT_PROMPT

    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")

    client = OpenAI(
        base_url=url,
        api_key=key,                             # → Authorization: Bearer (vLLM layer)
        default_headers={"X-API-Key": key},      # → platform serve-proxy layer
        timeout=float(os.getenv("PREDICT_TIMEOUT", "300")),
    )

    # 1) Readiness + model discovery. /v1/models doubles as a health check
    #    and frees us from hardcoding the model_id — after a checkpoint
    #    swap in serve_module.py this script keeps working untouched.
    try:
        models = [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {url} — deployment not RUNNING yet, or URL wrong: {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(
                f"[predict] {exc.status_code} from {url} — check "
                "RESON_INFERENCE_API_KEY (the job-scoped predict key from "
                "submit.py, not the platform rsk_ key)."
            )
        if exc.status_code == 404:
            sys.exit(
                f"[predict] 404 at {url}/models — endpoint is up but has no "
                "OpenAI surface. Was the job submitted with engine: "
                "vllm_openai in inference.yaml?"
            )
        raise

    if not models:
        sys.exit("[predict] /v1/models returned an empty list — replica still loading weights?")
    model = os.getenv("MODEL_ID", models[0])
    print(f"[predict] endpoint: {url}")
    print(f"[predict] models:   {models}  → using {model!r}")
    print(f"[predict] prompt:   {prompt}")
    print("─" * 72)

    # 2) Chat completion — streamed by default so you see tokens live,
    #    the way editor integrations consume this endpoint.
    messages = [
        {"role": "system", "content": "You are a concise, expert coding assistant."},
        {"role": "user", "content": prompt},
    ]

    if stream:
        chunks = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            stream=True,
        )
        for chunk in chunks:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
        )
        print(resp.choices[0].message.content)
        if resp.usage:
            print("─" * 72)
            print(
                f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
                f"{resp.usage.completion_tokens} completion"
            )


if __name__ == "__main__":
    main()
