"""
Deploy the Qwen3.5-122B-A10B level-2 text + vision assistant to ResonTech.

Ships this folder's ``inference.yaml`` (one file) and ``scripts/``
(recursively) via explicit keyword paths. No weights upload: vLLM pulls
the ~79 GB GPTQ-Int4 checkpoint from HF Hub on first boot (pre-stage it
on the host when you can; a cold pull takes tens of minutes).

Run
---
    cp .env.example .env  # fill in RESON_API_KEY (rsk_...) + S3 keys
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
        # Ordered GPU preference (>= 84 GB VRAM per card as configured, TP=1).
        # An RTX PRO 6000 96 GB host has no fleet label yet: add it here the
        # day one joins. For 2x80 GB TP=2 keep the H100/A100 entries and apply
        # the three edits in inference.yaml and serve_module.py.
        gpu_models=["h200", "h100_80gb", "a100_80gb"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "10.00")),
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Qwen3.5-122B-A10B Assistant + Vision"),
        inference=inference,
        inference_yaml=str(HERE / "inference.yaml"),
        scripts_dir=str(HERE / "scripts"),
    )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    print("[submit] first boot pulls ~79 GB from HF Hub; expect tens of minutes on a cold host")
    print("[submit] waiting for deployment to become RUNNING...")
    job.wait_until_ready(timeout=7200, poll=15)
    print(f"[submit] OK {job.id} is RUNNING")


if __name__ == "__main__":
    main()
