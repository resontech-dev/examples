"""
Classify a web page with the deployed BrowseSafe endpoint: does its HTML carry
instructions aimed at an AI agent?

Protocol from the model card: one user message whose content is the RAW HTML
(not extracted text), generation prompt on, answer is one token "yes" / "no".
Pages longer than the 16,384-token window are split into non-overlapping
chunks and the verdicts OR-ed: any "yes" flags the page.

Run
---
    pip install openai python-dotenv httpx
    # .env needs RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY (from submit.py)

    python predict.py                                  # sample_data/injected.html
    python predict.py sample_data/benign.html
    python predict.py https://example.com              # fetched with httpx, raw HTML
    python predict.py ./page.html --chunk-tokens 8000  # smaller chunks
"""
from __future__ import annotations

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

DEFAULT_PAGE = HERE / "sample_data" / "injected.html"
CHARS_PER_TOKEN = 4            # rough for HTML; the server tokenizes, this only sizes chunks
DEFAULT_CHUNK_TOKENS = 12000   # leaves room under the 16,384 window for the chat template


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


def _load_html(source: str) -> str:
    if source.startswith(("http://", "https://")):
        try:
            import httpx
        except ImportError:
            sys.exit("[predict] fetching a URL needs httpx: pip install httpx")
        resp = httpx.get(source, follow_redirects=True, timeout=30.0,
                         headers={"User-Agent": "reson-browsesafe-check/1.0"})
        resp.raise_for_status()
        return resp.text
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[predict] page not found: {path}")
    return path.read_text(errors="replace")


def _chunks(html: str, chunk_tokens: int) -> list[str]:
    size = chunk_tokens * CHARS_PER_TOKEN
    return [html[i:i + size] for i in range(0, len(html), size)] or [""]


def classify_chunk(client: OpenAI, model: str, html: str) -> tuple[bool, str]:
    """One chunk → (flagged, raw answer). Message layout verbatim from the card:
    the raw HTML as the user message, nothing else."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": html}],
        max_tokens=4,
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # VERIFY on the two sample pages: the card documents "yes" / "no" and the
    # chunking section calls the positive class VIOLATES, so "yes" is read as
    # "instructions for the agent present". Anything else is reported as-is.
    return raw.lower().startswith("yes"), raw


def main() -> None:
    argv = sys.argv[1:]
    chunk_tokens = DEFAULT_CHUNK_TOKENS
    if "--chunk-tokens" in argv:
        i = argv.index("--chunk-tokens")
        try:
            chunk_tokens = int(argv[i + 1])
        except (IndexError, ValueError):
            sys.exit("[predict] --chunk-tokens needs an integer")
        del argv[i:i + 2]
    source = argv[0] if argv else str(DEFAULT_PAGE)

    client = _client()
    models = _list_models(client)
    if not models:
        sys.exit("[predict] /v1/models returned an empty list: replica still loading weights?")
    model = os.getenv("MODEL_ID", "web-guard" if "web-guard" in models else models[0])
    html = _load_html(source)
    chunks = _chunks(html, chunk_tokens)
    print(f"[predict] models: {models} → using {model!r}")
    print(f"[predict] page:   {source} ({len(html)} chars → {len(chunks)} chunk(s) of ≤ {chunk_tokens} tokens)")
    print("─" * 72)

    flagged_any = False
    for n, chunk in enumerate(chunks, 1):
        t0 = time.perf_counter()
        flagged, raw = classify_chunk(client, model, chunk)
        dt = time.perf_counter() - t0
        flagged_any |= flagged
        print(f"[predict] chunk {n}/{len(chunks)}: {raw!r:<8} → {'FLAGGED' if flagged else 'clean'}  ({dt:.2f}s)")
        if raw.lower() not in ("yes", "no"):
            print("          ⚠️ answer is not yes/no: check the prompt format against the model card")
    print("─" * 72)
    if flagged_any:
        print("[predict] ✋ page contains instructions aimed at the agent: do not put it in the context")
    else:
        print("[predict] ✅ page is clean for the agent")


if __name__ == "__main__":
    main()
