"""
Deploy the BFSI document-ingest vision endpoint (Stack A) to ResonTech.

Ships this folder's ``inference.yaml`` (one file) and ``scripts/``
(recursively) via explicit keyword paths. No weights upload — vLLM pulls
the ~6 GB AWQ checkpoint from HF Hub on first boot.

⚠️ One-GPU pilot: Stack A (this job) and Stack B (the chat+embed bundle)
do NOT fit a 16 GB card together — stop one before starting the other.
See sectors/banking_and_insurance/bfsi_pilot/README.md, "Stack switching".

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

    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        auto_select_workers=True,
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "BFSI Docs Vision (Stack A)"),
        inference=inference,
        inference_yaml=str(HERE / "inference.yaml"),
        scripts_dir=str(HERE / "scripts"),
    )

    print()
    print("[submit] For the pilot's .env (bfsi_pilot):")
    print(f"    VL_BASE_URL = {job.endpoint}")
    print(f"    VL_API_KEY  = {job.predict_api_key}")
    print()

    print("[submit] first boot pulls the ~6 GB checkpoint from HF Hub")
    print("[submit] waiting for deployment to become RUNNING…")
    job.wait_until_ready(timeout=1800, poll=15)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
