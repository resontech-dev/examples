"""
Smoke-test a Gemma-3-family Ukrainian model (MamayLM / Lapa): text chat, optional image.

Run
---
    pip install openai python-dotenv
    python predict.py                                  # Ukrainian business reply (text)
    python predict.py "Стисло перекажи: <вставте текст документа>"
    python predict.py --image                          # sample_data/invoice.png → JSON
    python predict.py --image ./skan.png "Перепиши текст зі скану"
    python predict.py --stream "…"                     # token stream
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

SYSTEM = "Ти помічник працівника установи. Відповідай українською, стисло, діловим стилем, без вигаданих фактів."
DEFAULT_PROMPT = (
    "Підготуй відповідь на звернення мешканця: він просить пояснити, коли буде відремонтовано "
    "освітлення на вулиці Шевченка. Ремонт запланований на наступний тиждень, виконавець — КП «Благоустрій»."
)
DEFAULT_IMAGE = HERE / "sample_data" / "invoice.png"
DEFAULT_IMAGE_PROMPT = (
    "Витягни з цього рахунку: постачальника, номер, дату, валюту, позиції (назва, кількість, ціна, сума) "
    "і підсумок. Відповідай лише JSON з англійськими ключами."
)


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


def _image_part(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _run(client: OpenAI, model: str, messages: list, stream: bool, max_tokens: int, temperature: float) -> None:
    if stream:
        chunks = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        for chunk in chunks:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                print(delta, end="", flush=True)
        print()
        return
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
    )
    print(resp.choices[0].message.content)
    if resp.usage:
        print("─" * 72)
        print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion")


def main() -> None:
    argv = sys.argv[1:]
    stream = "--stream" in argv
    image = None
    if "--image" in argv:
        i = argv.index("--image")
        nxt = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
        is_img = bool(nxt) and (nxt.startswith("http") or Path(nxt).suffix.lower()
                                in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})
        image = nxt if is_img else str(DEFAULT_IMAGE)
        argv = [a for j, a in enumerate(argv) if j != i and not (is_img and j == i + 1)]
    positional = [a for a in argv if not a.startswith("--")]

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list — replica still loading weights?")
    model = os.getenv("MODEL_ID", "assistant" if "assistant" in models else models[0])
    print(f"[predict] models: {models} → using {model!r}")
    print("─" * 72)

    if image:
        print(f"[predict] image: {image}")
        prompt = positional[0] if positional else DEFAULT_IMAGE_PROMPT
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, _image_part(image)]}]
        _run(client, model, messages, stream, int(os.getenv("MAX_TOKENS", "768")), 0.0)
    else:
        prompt = positional[0] if positional else DEFAULT_PROMPT
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        _run(client, model, messages, stream, int(os.getenv("MAX_TOKENS", "512")), 0.3)


if __name__ == "__main__":
    main()
