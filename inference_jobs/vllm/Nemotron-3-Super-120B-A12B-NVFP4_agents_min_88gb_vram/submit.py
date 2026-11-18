"""
Deploy Nemotron 3 Super 120B-A12B (NVFP4) to ResonTech.

Needs the Ray 2.60.0 image on the head and workers (vLLM 0.29.0); see the
banner in scripts/serve_module.py.

Ships this folder's ``inference.yaml`` (one file) and ``scripts/``
(recursively) via explicit keyword paths. No weights upload: vLLM pulls
the 80.3 GB NVFP4 checkpoint from HF Hub on first boot. Pre-stage it on
the worker's NVMe for anything beyond a first test.

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

    # BYO mode: the YAML in this folder owns the scheduling declaration;
    # the scripts own runtime behavior. InferenceConfig only carries
    # submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        # Blackwell only (NVFP4), >= 88 GB per card. The target is the RTX
        # PRO 6000 96 GB, but the GPU catalog has no entry for it yet
        # (2026-11-18): add it first in this list once it does. Until then
        # B200 is the only listed card that fits. FP8 on 2x H100: README.
        gpu_models=["b200"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "10.00")),
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Nemotron 3 Super 120B NVFP4"),
        inference=inference,
        inference_yaml=str(HERE / "inference.yaml"),
        scripts_dir=str(HERE / "scripts"),
    )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    print("[submit] first boot pulls 80.3 GB from HF Hub and compiles kernels: expect 20+ minutes")
    print("[submit] waiting for deployment to become RUNNING…")
    job.wait_until_ready(timeout=3600, poll=15)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
