"""
Call the deployed the Mistral Small 24B multi-LoRA endpoint endpoint — base model and LoRA adapters.

Multi-LoRA serving: ONE base model in VRAM, N adapters loaded on demand
from ``lora_config.dynamic_lora_loading_path`` (an S3/GCS prefix). Clients
select an adapter per request via the ``model`` field:

    model="mistral-24b"                → base model, no adapter
    model="mistral-24b:<adapter_dir>"  → base + that adapter

where <adapter_dir> is a folder name under the S3 prefix containing a
standard PEFT LoRA checkpoint (adapter_config.json + adapter weights).

Run
---
    pip install openai python-dotenv
    python predict.py                          # base model
    python predict.py --adapter my-adapter     # base + adapter "my-adapter"
    python predict.py "custom prompt"
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


DEFAULT_PROMPT = "Draft a formal reply to a customer requesting an SLA credit for last week's outage."


def main() -> None:
    args = sys.argv[1:]
    adapter = None
    if "--adapter" in args:
        i = args.index("--adapter")
        try:
            adapter = args[i + 1]
        except IndexError:
            sys.exit("[predict] --adapter needs a name")
        args = args[:i] + args[i + 2:]
    prompt = args[0] if args else DEFAULT_PROMPT

    client = _client()
    models = _list_models(client)
    base = os.getenv("MODEL_ID", "mistral-24b")
    model = f"{base}:{adapter}" if adapter else base
    print(f"[predict] models: {models}")
    print(f"[predict] using:  {model!r}"
          + ("" if adapter else "  (pass --adapter <name> to hit a LoRA)"))
    print("─" * 72)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(os.getenv("MAX_TOKENS", "384")),
            temperature=0.7,
        )
    except APIStatusError as exc:
        if adapter and exc.status_code == 404:
            sys.exit(
                f"[predict] adapter {adapter!r} not found. The engine resolves "
                "adapters from lora_config.dynamic_lora_loading_path — upload a "
                f"PEFT checkpoint folder named {adapter!r} there and retry."
            )
        raise
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
