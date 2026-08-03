"""
Deploy the Qwen chat LLM job to ResonTech.

BYO mode (`from_files=THIS_DIR`) — the SDK ships this folder's
``inference.yaml`` and ``scripts/llm_serve.py`` as-is. No model weights
upload because the predictor pulls them from HuggingFace Hub on
``__init__`` (see ``init_args.model_id`` in inference.yaml).

Run
---
    cp .env.example .env  # fill in credentials
    python submit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceConfig, ResonTech, ResonTechConfig


HERE = Path(__file__).resolve().parent
os.chdir(HERE)
load_dotenv(HERE / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[submit] Missing required env var: {name}")
    return value


def main() -> None:
    sdk = ResonTech(
        ResonTechConfig(
            base_url=os.getenv("RESON_BASE_URL", "https://dev.api.beta.reson.tech"),
            s3_endpoint=os.getenv("RESON_S3_ENDPOINT", "https://s3.dev.beta.reson.tech"),
            platform_api_key=_required("RESON_API_KEY"),
            s3_access_key_id=_required("RESON_S3_KEY"),
            s3_secret_access_key=_required("RESON_S3_SECRET"),
        )
    )

    # BYO mode: the YAML in this folder owns model_id, init_args, cluster knobs,
    # and requirements. InferenceConfig only carries submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        auto_select_workers=True,
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Pipeline Parallel Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ"),
        inference=inference,
        # Explicit paths — local layout is whatever you want, the SDK puts
        # them in the right S3 slots. No weights file: the predictor pulls
        # Qwen from HuggingFace on __init__ (see init_args.model_id).
        inference_yaml=str(HERE / "inference.yaml"),
        scripts_dir=str(HERE / "scripts"),
    )

    print()
    print(f"[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    # First boot can take a while — the replica downloads Qwen2.5-0.5B
    # (~1 GB) from HuggingFace before serving. Bump the timeout accordingly.
    print("[submit] waiting for deployment to become RUNNING…")
    print("         (first boot pulls model weights from HF Hub — can be slow)")
    job.wait_until_ready(timeout=1800, poll=10)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
