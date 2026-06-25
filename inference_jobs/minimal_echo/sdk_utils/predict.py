"""
Call the deployed echo endpoint.

The predictor takes raw bytes (any text body) and returns::

    {"echo": "echo: <text>", "bytes_received": int,
     "request_index": int, "elapsed_ms": float}

Because the body is plain text (not structured JSON), we send it with
``client.predict(..., content_type="text/plain")`` — the whole body is
handed to ``predict(data: bytes)`` verbatim.

Run (from the job folder)
-------------------------
    # submit.py prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY — add them to .env, then:
    python sdk_utils/predict.py
    python sdk_utils/predict.py "hello world"
    python sdk_utils/predict.py "one" "two" "three"
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")   # .env lives in the job folder, next to inference.yaml

DEFAULT_MESSAGES = [
    "hello world",
    "the platform contract works",
]


def _collect_messages() -> list[str]:
    if len(sys.argv) > 1:
        return list(sys.argv[1:])
    return DEFAULT_MESSAGES


def main() -> None:
    client = InferenceClient.from_env()
    messages = _collect_messages()

    # ── A. Single-shot ──────────────────────────────────────────────────────
    first = messages[0]
    print(f"[predict] single → {first!r}")
    result = client.predict(first.encode("utf-8"), content_type="text/plain")
    print(f"        echo: {result.get('echo')!r}")
    print(
        f"        bytes_received={result.get('bytes_received')} "
        f"request_index={result.get('request_index')} "
        f"({result.get('elapsed_ms', 0):.2f} ms)"
    )

    # ── B. Batch — multiple messages in parallel ────────────────────────────
    if len(messages) > 1:
        print()
        print(f"[predict] batch  → {len(messages)} messages (concurrency=4)")
        results = client.predict_many(
            messages,
            concurrency=4,
            map_fn=lambda m: m.encode("utf-8"),
            on_error="collect",
        )
        for msg, r in zip(messages, results):
            if isinstance(r, Exception):
                print(f"  {msg[:30]:32} ✗ {r}")
                continue
            print(f"  {msg[:30]:32} → {r.get('echo')!r}")

    client.close()


if __name__ == "__main__":
    main()
