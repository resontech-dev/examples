"""
Call the deployed Qwen chat endpoint.

The predictor accepts JSON like::

    {"prompt": "...", "max_new_tokens": 64, "temperature": 0.7, "system": "..."}

…and returns::

    {"text": "...", "model_id": "...", "prompt_tokens": int,
     "completion_tokens": int, "generate_ms": float, ...}

⚠️ JSON payloads with parameters MUST be sent as ``application/octet-stream``
(``client.predict_json(...)``), NOT ``application/json``. The server's JSON
envelope only handles ``{"url": ...}`` / ``{"data_b64": ...}`` and silently
drops everything else — see ``InferenceClient.predict_json`` for the contract.

Run (from the job folder)
-------------------------
    # submit.py prints RESON_INFERENCE_URL + RESON_INFERENCE_API_KEY — add them to .env, then:
    python sdk_utils/predict.py
    python sdk_utils/predict.py "Explain attention in one sentence."
    python sdk_utils/predict.py "Pitch a dad joke." "Name a Mars rover." "Sum 17 + 24."
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceClient


HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")   # .env lives in the job folder, next to inference.yaml

# Generation knobs — edit freely.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

DEFAULT_PROMPTS = [
    "Explain attention in one sentence.",
    "What is the speed of light? Reply in 10 words or fewer.",
    "Write a one-line pitch for a CLI tool that lints YAML.",
]


def _collect_prompts() -> list[str]:
    if len(sys.argv) > 1:
        return list(sys.argv[1:])
    return DEFAULT_PROMPTS


def _format_text(text: str, width: int = 78) -> str:
    """Light pretty-print: indent multi-line, soft-wrap nothing else."""
    return "\n        ".join(text.splitlines() or [""])


def main() -> None:
    client = InferenceClient.from_env()
    prompts = _collect_prompts()

    # ── A. Single-shot ──────────────────────────────────────────────────────
    first = prompts[0]
    print(f"[predict] single → {first!r}")
    result = client.predict_json({
        "prompt": first,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    })
    if "error" in result:
        sys.exit(f"[predict] predictor returned an error: {result['error']}")
    print(f"        text: {_format_text(result['text'])}")
    print(
        f"        {result.get('completion_tokens', '?')} tokens in "
        f"{result.get('generate_ms', 0):.0f} ms "
        f"({result.get('throughput_tokens_per_s', 0):.1f} tok/s) "
        f"on {result.get('device', '?')}"
    )
    print(
        f"        replica: {client.last_response_headers.get('X-Replica-Id')}  "
        f"request_id: {client.last_response_headers.get('X-Request-Id')}"
    )

    # ── B. Batch — multiple prompts in parallel ────────────────────────────
    if len(prompts) > 1:
        print()
        print(f"[predict] batch  → {len(prompts)} prompts (concurrency=4)")

        # Each item turns into a JSON byte string. We do the encoding here
        # (in map_fn) so predict_many doesn't have to know about JSON.
        import json as _json

        def _encode(prompt: str) -> bytes:
            return _json.dumps({
                "prompt": prompt,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }).encode("utf-8")

        results = client.predict_many(
            prompts,
            concurrency=4,
            map_fn=_encode,
        )
        for prompt, r in zip(prompts, results):
            if isinstance(r, Exception):
                print(f"  {prompt[:40]:42} ✗ {r}")
                continue
            text = r.get("text") if isinstance(r, dict) else None
            tokens = r.get("completion_tokens", "?") if isinstance(r, dict) else "?"
            ms = r.get("generate_ms", 0) if isinstance(r, dict) else 0
            print(f"  {prompt[:40]:42} {ms:6.0f} ms ({tokens} tok)")
            print(f"      → {_format_text(text or '<no text>')}")

    client.close()


if __name__ == "__main__":
    main()
