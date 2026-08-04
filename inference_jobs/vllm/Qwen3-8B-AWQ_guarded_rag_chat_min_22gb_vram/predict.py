"""
Guarded RAG chat round trip against the deployed the guarded RAG chat bundle (Qwen3-8B + Qwen3Guard + embed + rerank) bundle.

One endpoint, four engines (see scripts/serve_module.py):
    "assistant" → the chat model
    "guard"     → Qwen3Guard-Gen-8B input/output moderation
    "embed"     → embeddings for retrieval
    "rerank"    → reranker (POST /v1/score, raw HTTP)

Flow per user message:
    1. guard(user input)      — refuse before spending chat tokens
    2. retrieve (embed + rerank over a toy corpus)
    3. chat with context
    4. guard(model output)    — belt & suspenders before returning

Run
---
    pip install openai python-dotenv httpx
    python predict.py
    python predict.py "What is our deployment freeze policy?"
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


import math

import httpx

CORPUS = [
    "Employees accrue 25 paid vacation days per calendar year, plus public holidays.",
    "Remote work is allowed up to 3 days per week after the probation period.",
    "Production deployments are frozen every Friday from 16:00 until Monday 08:00.",
    "All customer data is stored in the EU (Frankfurt region) and never leaves it.",
    "Support tickets marked SEV-1 must receive a first response within 30 minutes.",
    "The company pension plan matches employee contributions up to 4% of salary.",
]

DEFAULT_QUESTION = "When are production deployments frozen?"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _guard(client: OpenAI, text: str, role: str) -> tuple[bool, str]:
    """Ask Qwen3Guard-Gen for a safety verdict. Returns (is_safe, raw verdict).

    The guard is a generative classifier: give it the content, it emits a
    verdict line like "Safety: Safe" / "Safety: Unsafe" plus categories —
    exact output format is defined by the model card; we parse leniently.
    """
    resp = client.chat.completions.create(
        model="guard",
        messages=[{"role": "user", "content": f"[{role} message to moderate]\n{text}"}],
        max_tokens=64,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip()
    return "unsafe" not in verdict.lower(), verdict


def _retrieve(client: OpenAI, question: str, key: str) -> list[str]:
    emb = client.embeddings.create(model="embed", input=CORPUS + [question])
    vectors = [d.embedding for d in emb.data]
    doc_vecs, q_vec = vectors[:-1], vectors[-1]
    ranked = sorted(
        range(len(CORPUS)), key=lambda i: _cosine(doc_vecs[i], q_vec), reverse=True
    )[:4]
    top = [CORPUS[i] for i in ranked]
    try:
        resp = httpx.post(
            f"{str(client.base_url).rstrip('/')}/score",
            headers={"X-API-Key": key, "Authorization": f"Bearer {key}"},
            json={"model": "rerank", "text_1": question, "text_2": top},
            timeout=120.0,
        )
        if resp.status_code < 400:
            data = resp.json().get("data", [])
            order = sorted(((item.get("score", 0.0), i) for i, item in enumerate(data)),
                           reverse=True)
            top = [top[i] for _, i in order]
    except httpx.HTTPError:
        pass  # cosine order is a fine fallback
    return top[:3]


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    client = _client()
    key = _required("RESON_INFERENCE_API_KEY")
    models = _list_models(client)
    print(f"[predict] models: {models}")

    # 1) Guard the input
    safe, verdict = _guard(client, question, "user")
    print(f"[predict] input guard: {verdict!r}")
    if not safe:
        print("[predict] ✋ refused by input guard — not sending to the chat model.")
        return

    # 2) Retrieve
    context = _retrieve(client, question, key)
    print("[predict] context chunks:")
    for c in context:
        print(f"    • {c}")

    # 3) Chat
    resp = client.chat.completions.create(
        model="assistant",
        messages=[
            {"role": "system",
             "content": "Answer strictly from the provided context. If the answer "
                        "is not in the context, say you don't know."},
            {"role": "user",
             "content": "Context:\n" + "\n".join(f"- {c}" for c in context)
                        + f"\n\nQuestion: {question}"},
        ],
        max_tokens=int(os.getenv("MAX_TOKENS", "384")),
        temperature=0.2,
    )
    answer = resp.choices[0].message.content or ""

    # 4) Guard the output
    safe, verdict = _guard(client, answer, "assistant")
    print(f"[predict] output guard: {verdict!r}")
    print("─" * 72)
    print(answer if safe else "[predict] ✋ answer suppressed by output guard.")


if __name__ == "__main__":
    main()
