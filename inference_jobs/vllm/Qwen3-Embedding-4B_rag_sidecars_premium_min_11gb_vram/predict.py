"""
Retrieval round trip against the deployed Qwen3 4B RAG sidecars (embed + rerank).

One endpoint, two engines (see scripts/serve_module.py):
    "embed"   -> POST /v1/embeddings
    "rerank"  -> POST /v1/score      (raw HTTP; the openai lib has no score API)

Flow: embed a toy corpus plus the question, cosine top-4, rerank the top-4,
print both orders with scores and the wall time of each call. No chat call:
pair this job with any chat endpoint (the bundles use the same model ids).

Run
---
    pip install openai python-dotenv httpx
    python predict.py
    python predict.py "How many vacation days do employees get?"
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    sys.exit("[predict] The openai package is required: pip install openai")

try:
    import httpx
except ImportError:
    sys.exit("[predict] The httpx package is required: pip install httpx")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


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


def _rerank(base_url: str, key: str, query: str, docs: list[str],
            model: str = "rerank") -> list[tuple[float, int]] | None:
    """POST /v1/score -> [(score, doc index)] best first, or None if unavailable."""
    try:
        resp = httpx.post(
            f"{base_url}/score",
            headers={"X-API-Key": key, "Authorization": f"Bearer {key}"},
            json={"model": model, "text_1": query, "text_2": docs},
            timeout=120.0,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted(((item.get("score", 0.0), i) for i, item in enumerate(data)),
                      reverse=True)
    except httpx.HTTPError as exc:
        print(f"[predict] rerank unavailable ({exc})")
        return None


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    client = _client()
    key = _required("RESON_INFERENCE_API_KEY")
    models = _list_models(client)
    print(f"[predict] models: {models}")
    for required in ("embed", "rerank"):
        if required not in models:
            sys.exit(f"[predict] expected model_id {required!r} not served, check serve_module.py")
    print(f"[predict] question: {question}")
    print("-" * 72)

    # 1) Embed corpus + question, cosine top-4
    t0 = time.perf_counter()
    emb = client.embeddings.create(model="embed", input=CORPUS + [question])
    t_embed = time.perf_counter() - t0
    vectors = [d.embedding for d in emb.data]
    doc_vecs, q_vec = vectors[:-1], vectors[-1]
    scored = sorted(((_cosine(doc_vecs[i], q_vec), i) for i in range(len(CORPUS))),
                    reverse=True)[:4]
    print(f"[embed] {len(CORPUS) + 1} texts, dim {len(q_vec)}, {t_embed * 1000:.0f} ms. Cosine top-4:")
    for score, i in scored:
        print(f"    {score:.3f}  {CORPUS[i]}")
    top4 = [CORPUS[i] for _, i in scored]

    # 2) Rerank the top-4
    t0 = time.perf_counter()
    order = _rerank(str(client.base_url).rstrip("/"), key, question, top4)
    t_rerank = time.perf_counter() - t0
    if order is None:
        sys.exit("[predict] /v1/score returned 404: the rerank engine is not served")
    print(f"[rerank] {len(top4)} candidates, {t_rerank * 1000:.0f} ms. Reranked:")
    for score, i in order:
        print(f"    {score:.3f}  {top4[i]}")

    # 3) Optional third engine, if serve_module.py enables it
    if "rerank-mxbai" in models:
        t0 = time.perf_counter()
        alt = _rerank(str(client.base_url).rstrip("/"), key, question, top4, model="rerank-mxbai")
        t_alt = time.perf_counter() - t0
        if alt:
            print(f"[rerank-mxbai] {t_alt * 1000:.0f} ms. Reranked:")
            for score, i in alt:
                print(f"    {score:.3f}  {top4[i]}")


if __name__ == "__main__":
    main()
