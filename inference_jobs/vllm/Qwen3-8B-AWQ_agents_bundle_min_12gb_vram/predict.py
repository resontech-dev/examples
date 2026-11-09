"""
Smoke-test the agents bundle (Qwen3-8B + embed + rerank [+ guard]).

One endpoint, three or four engines (see scripts/serve_module.py):
    "assistant" → chat with tool calling; Qwen3 thinking off by default
    "embed"     → POST /v1/embeddings
    "rerank"    → POST /v1/score (raw HTTP — openai-python has no client for it)
    "guard"     → safety verdict, only when WITH_GUARD = True

Run
---
    pip install openai python-dotenv httpx
    python predict.py                         # chat + embed + rerank (+ guard), one pass
    python predict.py "Склади відповідь клієнту: рахунок №1042 оплачено двічі."
    python predict.py --tools                 # tool-calling round trip (hermes parser)
    python predict.py --think                 # leave Qwen3 thinking on (slower; the `reasoning` field)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import httpx
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
CORPUS = [
    "Повернення коштів за помилкові списання виконується протягом 3 робочих днів.",
    "Доставка по Києву безкоштовна від 1 000 грн, по Україні — за тарифами перевізника.",
    "Гарантія на холодильне обладнання — 24 місяці з дати встановлення.",
    "Графік роботи підтримки: пн–пт 9:00–18:00.",
]
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
    """Per-request thinking switch. The server has no default: Qwen3 thinks unless told not to."""
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enabled}}}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def step_guard(client: OpenAI, text: str) -> None:
    resp = client.chat.completions.create(
        model="guard",
        messages=[{"role": "user", "content": text}],
        max_tokens=48,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().replace("\n", " | ")
    print(f"[guard]     {verdict}")


def step_chat(client: OpenAI, prompt: str, think: bool) -> None:
    resp = client.chat.completions.create(
        model="assistant",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=int(os.getenv("MAX_TOKENS", "384")),
        temperature=0.2,
        **_think(think),
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        print(f"[assistant] reasoning: {len(reasoning)} chars (thinking was on)")
    print(f"[assistant] {msg.content}")
    if resp.usage:
        print(f"[assistant] tokens: {resp.usage.prompt_tokens} prompt + "
              f"{resp.usage.completion_tokens} completion")


def step_retrieve(client: OpenAI, question: str, key: str) -> None:
    emb = client.embeddings.create(model="embed", input=CORPUS + [question])
    vectors = [d.embedding for d in emb.data]
    doc_vecs, q_vec = vectors[:-1], vectors[-1]
    ranked = sorted(range(len(CORPUS)), key=lambda i: _cosine(doc_vecs[i], q_vec), reverse=True)
    print(f"[embed]     dim={len(q_vec)}; cosine top-1: {CORPUS[ranked[0]]!r}")

    resp = httpx.post(
        f"{str(client.base_url).rstrip('/')}/score",
        headers={"X-API-Key": key, "Authorization": f"Bearer {key}"},
        json={"model": "rerank", "text_1": question, "text_2": CORPUS},
        timeout=120.0,
    )
    if resp.status_code >= 400:
        print(f"[rerank]    HTTP {resp.status_code}: {resp.text[:200]}")
        return
    scores = [item.get("score", 0.0) for item in resp.json().get("data", [])]
    best = max(range(len(CORPUS)), key=lambda i: scores[i]) if scores else None
    if best is not None:
        print(f"[rerank]    top-1 (score {scores[best]:.3f}): {CORPUS[best]!r}")


def step_tools(client: OpenAI, think: bool) -> None:
    question = "Який статус замовлення 10452?"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    first = client.chat.completions.create(
        model="assistant", messages=messages, tools=TOOLS, tool_choice="auto",
        max_tokens=256, temperature=0.0, **_think(think),
    )
    msg = first.choices[0].message
    call = (msg.tool_calls or [None])[0]
    if call is None:
        sys.exit(f"[tools] no tool call — the model answered with text: {msg.content!r}")
    print(f"[tools]     call: {call.function.name}({call.function.arguments})")

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
        model="assistant", messages=messages, tools=TOOLS,
        max_tokens=256, temperature=0.2, **_think(think),
    )
    print(f"[tools]     answer: {second.choices[0].message.content}")


def main() -> None:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prompt = args[0] if args else DEFAULT_PROMPT
    think = "--think" in flags

    client = _client()
    key = _required("RESON_INFERENCE_API_KEY")
    models = _list_models(client)
    print(f"[predict]   models: {models}")
    print("─" * 72)

    if "guard" in models:
        step_guard(client, prompt)
    step_chat(client, prompt, think)
    step_retrieve(client, prompt, key)
    if "--tools" in flags:
        step_tools(client, think)


if __name__ == "__main__":
    main()
