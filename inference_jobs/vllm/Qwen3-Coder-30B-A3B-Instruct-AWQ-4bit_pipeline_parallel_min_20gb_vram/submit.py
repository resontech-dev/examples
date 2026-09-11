"""
Deploy the Qwen3-Coder-30B pipeline-parallel vLLM job to ResonTech.

Ships this folder's ``inference.yaml`` (one file) and ``scripts/``
(recursively) via explicit keyword paths. No weights upload — vLLM pulls
the AWQ checkpoint (~18 GB) from HuggingFace Hub on first boot, and with
PP=2 **both** workers download the full snapshot (each loads only its
layers but fetches everything).

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
        # Ordered GPU preference (>= 20 GB VRAM total, sharded across cards); first entry is filled first.
        gpu_models=["rtx_4090", "rtx_3090", "rtx_a5000"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "2.50")),
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Qwen3-Coder-30B PP=2"),
        inference=inference,
        inference_yaml=str(HERE / "inference.yaml"),
        scripts_dir=str(HERE / "scripts"),
    )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    # First boot: BOTH workers download the ~18 GB AWQ checkpoint from
    # HF Hub (PP stages fetch the full snapshot), then the engine
    # pre-allocates VRAM. Expect many minutes on a cold host.
    print("[submit] waiting for deployment to become RUNNING…")
    print("         (first boot pulls ~18 GB on each PP worker — be patient)")
    job.wait_until_ready(timeout=3600, poll=15)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
