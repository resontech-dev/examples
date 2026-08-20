"""
Talk to the deployed the Qwen3-Coder-480B frontier coding endpoint (TEMPLATE) endpoint (OpenAI protocol).

Auth crosses two layers and this script satisfies both:
  * platform edge (serve-proxy nginx) checks ``X-API-Key``  → default_headers
  * vLLM engine may check ``Authorization: Bearer``         → api_key=

Run
---
    pip install openai python-dotenv
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                          # default prompt, streamed
    python predict.py "your prompt here"
    python predict.py --no-stream "…"
    python predict.py --tools                  # function-calling smoke test
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


DEFAULT_PROMPT = "Refactor a 2k-line God class into cohesive services. Outline the plan, then show the first extraction."

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def _tools_demo(client: OpenAI, model: str) -> None:
    """One round-trip of OpenAI function calling against the endpoint."""
    print("[predict] tools demo → asking about the weather in Vienna…")
    messages = [{"role": "user", "content": "What's the weather in Vienna right now?"}]
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=[WEATHER_TOOL], max_tokens=256,
    )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        print("[predict] model answered without a tool call (parser off or model chose not to):")
        print(f"          {msg.content}")
        return
    call = msg.tool_calls[0]
    print(f"[predict] tool call: {call.function.name}({call.function.arguments})")
    messages += [
        msg.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id,
         "content": '{"city": "Vienna", "temp_c": 21, "conditions": "clear"}'},
    ]
    final = client.chat.completions.create(model=model, messages=messages, max_tokens=256)
    print(f"[predict] final answer: {final.choices[0].message.content}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stream = "--no-stream" not in sys.argv
    prompt = args[0] if args else DEFAULT_PROMPT

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list — replica still loading weights?")
    model = os.getenv("MODEL_ID", models[0])
    print(f"[predict] endpoint: {client.base_url}")
    print(f"[predict] models:   {models}  → using {model!r}")

    if "--tools" in sys.argv:
        _tools_demo(client, model)
        return

    print(f"[predict] prompt:   {prompt}")
    print("─" * 72)
    messages = [
        {"role": "system", "content": "You are a concise, expert coding assistant."},
        {"role": "user", "content": prompt},
    ]

    if stream:
        chunks = client.chat.completions.create(
            model=model, messages=messages,
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
            model=model, messages=messages,
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
        )
        print(resp.choices[0].message.content)
        if resp.usage:
            print("─" * 72)
            print(f"[predict] tokens: {resp.usage.prompt_tokens} prompt + "
                  f"{resp.usage.completion_tokens} completion")


if __name__ == "__main__":
    main()
