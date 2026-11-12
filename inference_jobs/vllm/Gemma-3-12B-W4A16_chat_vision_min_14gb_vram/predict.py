"""
Smoke-test Gemma 3 12B (W4A16): text chat in English or Ukrainian, optional image.

Run
---
    pip install openai python-dotenv
    python predict.py                                  # English business reply
    python predict.py --uk                             # Ukrainian business reply
    python predict.py "your prompt here"
    python predict.py --image                          # sample_data/invoice.png -> JSON
    python predict.py --image ./scan.png "Transcribe this scan verbatim"
    python predict.py --long                           # ~10k-token prompt, see README "Checkpoint"
    python predict.py --stream "..."                   # token stream
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

SYSTEM_EN = "You are a concise, professional assistant. Do not invent facts."
SYSTEM_UK = "Ти помічник працівника установи. Відповідай українською, стисло, діловим стилем, без вигаданих фактів."
PROMPT_EN = (
    "A customer writes: 'I was charged twice for invoice 1042.' Draft a short, polite reply: the duplicate charge will be refunded within 3 business days."
)
PROMPT_UK = (
    "Підготуй відповідь на звернення мешканця: він просить пояснити, коли буде відремонтовано освітлення на вулиці Шевченка. Ремонт запланований на наступний тиждень, виконавець: КП «Благоустрій»."
)
DEFAULT_IMAGE = HERE / "sample_data" / "invoice.png"
IMAGE_PROMPT = (
    "Extract from this invoice: vendor, invoice_number, date, currency, line_items (description, qty, unit_price, amount) and total. Reply with JSON only."
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


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


def _image_part(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _take_path_flag(argv: list[str], flag: str, suffixes: set[str], default: Path) -> tuple[str | None, list[str]]:
    """Pop `--flag [path]` from argv. Returns (path or None, remaining argv)."""
    if flag not in argv:
        return None, argv
    i = argv.index(flag)
    nxt = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
    is_path = bool(nxt) and (nxt.startswith("http") or Path(nxt).suffix.lower() in suffixes)
    value = nxt if is_path else str(default)
    rest = [a for j, a in enumerate(argv) if j != i and not (is_path and j == i + 1)]
    return value, rest


def _long_prompt() -> str:
    """~10k tokens with a fact at the start and the question at the end.

    The Red Hat tokenizer.json carries a right-side truncation default of 8192
    tokens. If the server ever applied it, the question at the end would be cut
    and the answer would miss the code; prompt_tokens would also stop at 8192.
    """
    filler = "".join(
        f"Section {i}: the committee reviewed routine maintenance reports and approved "
        "the schedule without changes. "
        for i in range(1, 451)
    )
    return (
        "Note for later: the access code for the archive room is 4719.\n\n"
        + filler
        + "\n\nQuestion: what is the access code for the archive room mentioned at the very start? "
        "Answer with the number only."
    )


def _run(client: OpenAI, model: str, messages: list, stream: bool, max_tokens: int, temperature: float) -> None:
    started = time.perf_counter()
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
        print("-" * 72)
        print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion, {time.perf_counter() - started:.1f} s")


def main() -> None:
    argv = sys.argv[1:]
    stream = "--stream" in argv
    ukrainian = "--uk" in argv
    long_check = "--long" in argv
    image, argv = _take_path_flag(argv, "--image", IMAGE_SUFFIXES, DEFAULT_IMAGE)
    positional = [a for a in argv if not a.startswith("--")]

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list: replica still loading weights?")
    model = os.getenv("MODEL_ID", "assistant" if "assistant" in models else models[0])
    print(f"[predict] models: {models} -> using {model!r}")
    print("-" * 72)

    if image:
        print(f"[predict] image: {image}")
        prompt = positional[0] if positional else IMAGE_PROMPT
        # Gemma: image before text in the same message.
        messages = [{"role": "user", "content": [_image_part(image), {"type": "text", "text": prompt}]}]
        _run(client, model, messages, stream, int(os.getenv("MAX_TOKENS", "768")), 0.0)
    elif long_check:
        messages = [{"role": "user", "content": _long_prompt()}]
        print("[predict] long-prompt check: expect '4719' and more than 8192 prompt tokens")
        _run(client, model, messages, False, 16, 0.0)
    else:
        system = SYSTEM_UK if ukrainian else SYSTEM_EN
        prompt = positional[0] if positional else (PROMPT_UK if ukrainian else PROMPT_EN)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        _run(client, model, messages, stream, int(os.getenv("MAX_TOKENS", "512")), 0.3)


if __name__ == "__main__":
    main()
