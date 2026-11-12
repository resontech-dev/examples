"""
Smoke-test the Qwen3.5-122B-A10B text + vision assistant (one model id for both).

Run
---
    pip install openai python-dotenv
    python predict.py                                  # Ukrainian support reply (text)
    python predict.py "Класифікуй лист: 'Коли приїде майстер?' → тип + терміновість, JSON."
    python predict.py --image                          # sample_data/invoice.png → JSON
    python predict.py --image ./scan.png "Перелічи всі позиції рахунку"
    python predict.py --image https://example.com/receipt.jpg
    python predict.py --tools                          # tool-calling round trip
    python predict.py --think "…"                      # leave thinking on (the `reasoning` field)
"""
from __future__ import annotations

import base64
import json
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

SYSTEM = "Ти асистент служби підтримки. Відповідай стисло, українською, без вигаданих фактів."
DEFAULT_PROMPT = (
    "Клієнт пише: «Мені двічі списали кошти за рахунок №1042». Напиши коротку ввічливу "
    "відповідь: дублікат повернемо протягом 3 робочих днів."
)
DEFAULT_IMAGE = HERE / "sample_data" / "invoice.png"
DEFAULT_IMAGE_PROMPT = (
    "Extract from this invoice: vendor, invoice_number, date, currency, "
    "line_items (description, qty, unit_price, amount) and total. Reply with JSON only."
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Повертає статус замовлення за його номером.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Номер замовлення"}},
            "required": ["order_id"],
        },
    },
}]


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


def _think(enabled: bool) -> dict:
    """Per-request thinking switch. The server has no default: the model thinks unless told not to."""
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enabled}}}


def _image_part(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _print(resp) -> None:
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        print(f"[assistant] reasoning: {len(reasoning)} chars (thinking was on)")
    print(msg.content)
    if resp.usage:
        print("─" * 72)
        print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion")


def run_text(client: OpenAI, model: str, prompt: str, think: bool) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=int(os.getenv("MAX_TOKENS", "384")),
        temperature=0.2,
        **_think(think),
    )
    _print(resp)


def run_image(client: OpenAI, model: str, image: str, prompt: str, think: bool) -> None:
    print(f"[predict] image: {image}")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, _image_part(image)]}],
        max_tokens=int(os.getenv("MAX_TOKENS", "768")),
        temperature=0.0,
        **_think(think),
    )
    _print(resp)


def run_tools(client: OpenAI, model: str, think: bool) -> None:
    question = "Який статус замовлення 10452?"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    first = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto",
        max_tokens=256, temperature=0.0, **_think(think),
    )
    msg = first.choices[0].message
    call = (msg.tool_calls or [None])[0]
    if call is None:
        sys.exit(f"[tools] no tool call — the model answered with text: {msg.content!r}")
    print(f"[tools] call: {call.function.name}({call.function.arguments})")

    args = json.loads(call.function.arguments or "{}")
    result = {"order_id": args.get("order_id"), "status": "в дорозі", "eta": "завтра до 14:00"}
    messages += [
        {"role": "assistant", "content": msg.content or "",
         "tool_calls": [{"id": call.id, "type": "function",
                         "function": {"name": call.function.name,
                                      "arguments": call.function.arguments}}]},
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)},
    ]
    second = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS,
        max_tokens=256, temperature=0.2, **_think(think),
    )
    print(f"[tools] answer: {second.choices[0].message.content}")


def main() -> None:
    argv = sys.argv[1:]
    think = "--think" in argv
    tools = "--tools" in argv
    image = None
    if "--image" in argv:
        i = argv.index("--image")
        nxt = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else None
        image = nxt if (nxt and (nxt.startswith("http") or Path(nxt).suffix.lower()
                                 in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})) \
            else str(DEFAULT_IMAGE)
        argv = [a for j, a in enumerate(argv) if j != i and not (nxt and a == nxt and j == i + 1)]
    positional = [a for a in argv if not a.startswith("--")]

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list — replica still loading weights?")
    model = os.getenv("MODEL_ID", "assistant" if "assistant" in models else models[0])
    print(f"[predict] models: {models} → using {model!r}")
    print("─" * 72)

    if tools:
        run_tools(client, model, think)
    elif image:
        run_image(client, model, image, positional[0] if positional else DEFAULT_IMAGE_PROMPT, think)
    else:
        run_text(client, model, positional[0] if positional else DEFAULT_PROMPT, think)


if __name__ == "__main__":
    main()
