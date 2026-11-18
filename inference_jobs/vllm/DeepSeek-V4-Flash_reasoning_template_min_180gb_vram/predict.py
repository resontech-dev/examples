"""
Smoke-test the DeepSeek V4-Flash (TEMPLATE) endpoint: a chat and a tool-call round trip.

Thinking is off by default on DeepSeek V4; --think turns it on at effort
"high" through chat_template_kwargs.

Run
---
    pip install openai python-dotenv
    python predict.py                                  # Ukrainian support reply
    python predict.py "Класифікуй лист: 'Коли приїде майстер?' → тип + терміновість, JSON."
    python predict.py --tools                          # tool-calling round trip
    python predict.py --think "…"                      # thinking on, reasoning_effort high (the `reasoning` field)
"""
from __future__ import annotations

import json
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

MODEL_ID = "deepseek-v4-flash"
SYSTEM = "Ти асистент служби підтримки. Відповідай стисло, українською, без вигаданих фактів."
DEFAULT_PROMPT = (
    "Клієнт пише: «Мені двічі списали кошти за рахунок №1042». Напиши коротку ввічливу "
    "відповідь: дублікат повернемо протягом 3 робочих днів."
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


def _think(enabled: bool) -> dict:
    """DeepSeek V4 switch: off by default; effort low / high / max when on."""
    kwargs = {"thinking": True, "reasoning_effort": "high"} if enabled else {"thinking": False}
    return {"extra_body": {"chat_template_kwargs": kwargs}}


def _print(resp) -> None:
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
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
        max_tokens=int(os.getenv("MAX_TOKENS", "4096" if think else "512")),
        temperature=1.0 if think else 0.2,
        **_think(think),
    )
    _print(resp)


def run_tools(client: OpenAI, model: str, think: bool) -> None:
    question = "Який статус замовлення 10452?"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    first = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, tool_choice="auto",
        max_tokens=4096 if think else 512, temperature=1.0 if think else 0.2, **_think(think),
    )
    msg = first.choices[0].message
    call = (msg.tool_calls or [None])[0]
    if call is None:
        sys.exit(f"[tools] no tool call, the model answered with text: {msg.content!r}")
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
        max_tokens=4096 if think else 512, temperature=1.0 if think else 0.2, **_think(think),
    )
    print(f"[tools] answer: {second.choices[0].message.content}")


def main() -> None:
    argv = sys.argv[1:]
    think = "--think" in argv
    tools = "--tools" in argv
    positional = [a for a in argv if not a.startswith("--")]

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list: replica still loading weights?")
    model = os.getenv("MODEL_ID", MODEL_ID if MODEL_ID in models else models[0])
    print(f"[predict] models: {models} → using {model!r}")
    print("─" * 72)

    if tools:
        run_tools(client, model, think)
    else:
        run_text(client, model, positional[0] if positional else DEFAULT_PROMPT, think)


if __name__ == "__main__":
    main()
