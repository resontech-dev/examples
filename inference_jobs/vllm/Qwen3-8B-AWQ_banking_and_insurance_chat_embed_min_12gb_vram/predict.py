"""
Smoke-test the BFSI agent bundle (Stack B) — chat + embeddings.

The real client is sectors/banking_and_insurance/bfsi_pilot/agent_server.py;
this script proves both engines answer.

Run
---
    pip install openai python-dotenv
    python predict.py                # chat smoke (streamed) + embed dim check
    python predict.py "your prompt"
    python predict.py --tools        # hermes function-calling round trip
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

DEFAULT_PROMPT = "Поясни, що таке франшиза в страховому полісі, у двох реченнях."

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_claim_status",
        "description": "Get the status of an insurance claim by id.",
        "parameters": {"type": "object",
                       "properties": {"claim_id": {"type": "string"}},
                       "required": ["claim_id"]},
    },
}


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[predict] Missing required env var: {name} (see submit.py output)")
    return value


def _openai_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/predict"):
        base = base[: -len("/predict")]
    return f"{base}/v1"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prompt = args[0] if args else DEFAULT_PROMPT

    url = _openai_base_url(_required("RESON_INFERENCE_URL"))
    key = _required("RESON_INFERENCE_API_KEY")
    client = OpenAI(base_url=url, api_key=key,
                    default_headers={"X-API-Key": key},
                    timeout=float(os.getenv("PREDICT_TIMEOUT", "300")))

    try:
        models = [m.id for m in client.models.list()]
    except APIConnectionError as exc:
        sys.exit(f"[predict] Cannot reach {url} — deployment not RUNNING yet? {exc}")
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            sys.exit(f"[predict] {exc.status_code} — check RESON_INFERENCE_API_KEY")
        raise
    print(f"[predict] models: {models}")

    # Embeddings engine check — the pilot's pgvector column is vector(1024).
    if "embed" in models:
        emb = client.embeddings.create(model="embed", input=["франшиза 5000 грн"])
        dim = len(emb.data[0].embedding)
        print(f"[predict] embed OK — dim={dim} " + ("✔" if dim == 1024 else "⚠ expected 1024"))

    if "--tools" in sys.argv:
        resp = client.chat.completions.create(
            model="assistant",
            messages=[{"role": "user", "content": "Який статус клейму CLM-0007?"}],
            tools=[WEATHER_TOOL], max_tokens=256,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            c = msg.tool_calls[0]
            print(f"[predict] tool call ✔ {c.function.name}({c.function.arguments})")
        else:
            print(f"[predict] no tool call — parser off? Answer: {msg.content}")
        return

    print(f"[predict] prompt: {prompt}")
    print("─" * 72)
    stream = client.chat.completions.create(
        model="assistant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=384, temperature=0.2, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
