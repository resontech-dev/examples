"""
Send an image to the deployed the Qwen2.5-VL-7B document-vision endpoint endpoint (OpenAI vision protocol).

Images travel as standard OpenAI content parts — a data URL for local
files, or a plain URL the replica can reach. Default input is the synthetic
invoice in sample_data/; the prompt asks for structured JSON, which is the
document-understanding workhorse use case.

Run
---
    pip install openai python-dotenv
    python predict.py                            # sample invoice → JSON
    python predict.py ./scan.png "List all line items"
    python predict.py https://example.com/receipt.jpg
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


import base64
import mimetypes

DEFAULT_IMAGE = HERE / "sample_data" / "invoice.png"
DEFAULT_PROMPT = (
    "Extract from this invoice: vendor, invoice_number, date, currency, "
    "line_items (description, qty, unit_price, amount) and total. "
    "Reply with JSON only."
)


def _image_part(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def main() -> None:
    args = sys.argv[1:]
    image = args[0] if args else str(DEFAULT_IMAGE)
    prompt = args[1] if len(args) > 1 else DEFAULT_PROMPT

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list — replica still loading weights?")
    model = os.getenv("MODEL_ID", models[0])
    print(f"[predict] models: {models} → using {model!r}")
    print(f"[predict] image:  {image}")
    print("─" * 72)

    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}, _image_part(image)],
        }],
        max_tokens=int(os.getenv("MAX_TOKENS", "768")),
        temperature=0.0,
    )
    print(resp.choices[0].message.content)
    if resp.usage:
        print("─" * 72)
        print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion")


if __name__ == "__main__":
    main()
