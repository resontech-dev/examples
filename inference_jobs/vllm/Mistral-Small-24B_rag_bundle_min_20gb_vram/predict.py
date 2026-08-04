"""
Mini-RAG round trip against the deployed the Mistral Small 24B RAG bundle (chat+embed+rerank) bundle.

One endpoint, three engines (see scripts/serve_module.py):
    "assistant"  → POST /v1/chat/completions
    "embed"        → POST /v1/embeddings
    "rerank"       → POST /v1/score      (raw HTTP — openai lib has no score API)

Flow: embed a toy corpus → cosine top-5 → rerank top-5 → answer with the
top-3 chunks as context. This is the wiring customers copy into their RAG
stack; swap the corpus for their chunked documents.

Run
---
    pip install openai python-dotenv httpx
    python predict.py
    python predict.py "How many vacation days do employees get?"
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


import json
import math

import httpx

CORPUS = [
    "Employees accrue 25 paid vacation days per calendar year, plus public holidays.",
    "Remote work is allowed up to 3 days per week after the probation period.",
    "The probation period for new hires is 6 months with a mid-point review.",
    "Travel expenses are reimbursed within 30 days when filed through the portal.",
    "The company pension plan matches employee contributions up to 4% of salary.",
    "Support tickets marked SEV-1 must receive a first response within 30 minutes.",
    "Production deployments are frozen every Friday from 16:00 until Monday 08:00.",
    "All customer data is stored in the EU (Frankfurt region) and never leaves it.",
]

DEFAULT_QUESTION = "How many vacation days do I get, and can I work remotely?"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _rerank(base_url: str, key: str, query: str, docs: list[str]) -> list[int] | None:
    """POST /v1/score → ranked doc indices (best first), or None if unavailable."""
    try:
        resp = httpx.post(
            f"{base_url}/score",
            headers={"X-API-Key": key, "Authorization": f"Bearer {key}"},
            json={"model": "rerank", "text_1": query, "text_2": docs},
            timeout=120.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json().get("data", [])
        scored = sorted(
            ((item.get("score", 0.0), i) for i, item in enumerate(data)), reverse=True
        )
        return [i for _, i in scored]
    except httpx.HTTPError as exc:
        print(f"[predict] rerank unavailable ({exc}) — falling back to cosine order")
        return None


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    client = _client()
    models = _list_models(client)
    print(f"[predict] models: {models}")
    chat_id = os.getenv("CHAT_MODEL_ID", "assistant")
    for required in (chat_id, "embed"):
        if required not in models:
            sys.exit(f"[predict] expected model_id {required!r} not served — check serve_module.py")

    # 1) Embed corpus + query, cosine top-5
    print(f"[predict] embedding {len(CORPUS)} chunks + query…")
    emb = client.embeddings.create(model="embed", input=CORPUS + [question])
    vectors = [d.embedding for d in emb.data]
    doc_vecs, q_vec = vectors[:-1], vectors[-1]
    ranked = sorted(
        range(len(CORPUS)), key=lambda i: _cosine(doc_vecs[i], q_vec), reverse=True
    )[:5]
    top5 = [CORPUS[i] for i in ranked]

    # 2) Rerank top-5 → top-3 (the cheapest RAG quality win)
    order = _rerank(str(client.base_url).rstrip("/"), _required("RESON_INFERENCE_API_KEY"),
                    question, top5)
    context = [top5[i] for i in (order or range(len(top5)))[:3]]
    print("[predict] context chunks:")
    for c in context:
        print(f"    • {c}")

    # 3) Answer with context
    print("─" * 72)
    chunks = client.chat.completions.create(
        model=chat_id,
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
        stream=True,
    )
    for chunk in chunks:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
