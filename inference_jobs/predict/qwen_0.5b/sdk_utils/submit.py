"""
Deploy the Qwen chat LLM job to ResonTech.

BYO mode — point the SDK at this job's ``inference.yaml`` (one file) and
``scripts/`` (a directory, copied to storage recursively). No ``model_file``
because the predictor pulls weights from HuggingFace Hub on ``__init__``
(see ``init_args.model_id`` in inference.yaml).

Run (from the job folder)
-------------------------
    cp .env.example .env          # then fill in your credentials
    python sdk_utils/submit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceConfig, ResonTech, ResonTechConfig


HERE = Path(__file__).resolve().parent     # .../qwen_0.5b/sdk_utils
ROOT = HERE.parent                          # .../qwen_0.5b  (holds inference.yaml + scripts/)
os.chdir(ROOT)
load_dotenv(ROOT / ".env")


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
    print("[submit] authed with platform api key")

    # BYO mode: the job's inference.yaml owns model_id, init_args, cluster
    # knobs, and requirements. InferenceConfig only carries submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        # Ordered GPU preference (>= 12 GB VRAM per card); first entry is filled first.
        gpu_models=["rtx_5070", "rtx_4070", "rtx_3060"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "2.50")),
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Qwen 0.5B Chat"),
        inference=inference,
        inference_yaml=str(ROOT / "inference.yaml"),   # REQUIRED — one file
        scripts_dir=str(ROOT / "scripts"),             # REQUIRED — dir, copied recursively
        # model_file / sample_data_dir not needed: weights pull from HF Hub.
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
