"""
Deploy the Qwen chat LLM job to ResonTech.

BYO mode (`from_files=JOB_ROOT`) — the SDK ships the job folder (the
parent of this sdk_utils/ dir): its ``inference.yaml`` and
``scripts/llm_serve.py`` as-is. No model weights upload because the
predictor pulls them from HuggingFace Hub on ``__init__`` (see
``init_args.model_id`` in inference.yaml).

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
            email=_required("RESON_EMAIL"),
            password=_required("RESON_PASSWORD"),
            s3_access_key_id=_required("RESON_S3_KEY"),
            s3_secret_access_key=_required("RESON_S3_SECRET"),
        )
    )
    user = sdk.login()
    print(f"[submit] logged in as {user.email}")

    # BYO mode: the job's inference.yaml owns model_id, init_args, cluster
    # knobs, and requirements. InferenceConfig only carries submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        auto_select_workers=True,
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Qwen 0.5B Chat"),
        inference=inference,
        from_files=str(ROOT),         # ship the job folder as-is, skip generation
    )

    print()
    print(f"[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.api_key}")
    print()

    # First boot can take a while — the replica downloads Qwen2.5-0.5B
    # (~1 GB) from HuggingFace before serving. Bump the timeout accordingly.
    print("[submit] waiting for deployment to become RUNNING…")
    print("         (first boot pulls model weights from HF Hub — can be slow)")
    job.wait_until_ready(timeout=1800, poll=10)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
