"""
Deploy the Qwen2.5-Coder-7B data-parallel vLLM job to ResonTech.

Ships this folder's ``inference.yaml`` (one file) and ``scripts/``
(recursively) via explicit keyword paths. No weights upload — vLLM pulls
the AWQ checkpoint (~4.9 GB) from HuggingFace Hub on each worker's first
boot (2 workers → 2 downloads).

Run
---
    cp .env.example .env  # fill in RESON_API_KEY (rsk_…) + S3 keys
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

    # BYO mode: the YAML in this folder owns the cluster/vllm declaration;
    # serve_module.py owns runtime behavior. InferenceConfig only carries
    # submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        # Ordered GPU preference (>= 6 GB VRAM per card); first entry is filled first.
        gpu_models=["rtx_2080ti", "rtx_5070", "rtx_4070"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "2.50")),
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Qwen2.5-Coder-7B DPx2"),
        inference=inference,
        inference_yaml=str(HERE / "inference.yaml"),
        scripts_dir=str(HERE / "scripts"),
    )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    # First boot: each of the 2 workers downloads the ~4.9 GB AWQ
    # checkpoint from HF Hub, then the engine pre-allocates VRAM.
    print("[submit] waiting for deployment to become RUNNING…")
    print("         (first boot pulls the checkpoint from HF Hub — minutes)")
    job.wait_until_ready(timeout=1800, poll=10)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
