"""
Deploy the image-classification predictor to ResonTech via the SDK.

BYO mode (`from_files=JOB_ROOT`) — the SDK ships the job folder (the
parent of this sdk_utils/ dir): its ``inference.yaml``,
``scripts/classifier_serve.py``, and ``model/`` as-is.

If you drop a trained checkpoint at ``model/classifier.pt`` it ships with
the bundle and ``load_weights()`` mounts it. If ``model/`` is empty (the
default), the predictor falls back to ImageNet-pretrained ResNet50 so the
deploy still answers requests — you'll just see ImageNet labels.

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


HERE = Path(__file__).resolve().parent     # .../image_classification/sdk_utils
ROOT = HERE.parent                          # .../image_classification  (holds inference.yaml + scripts/)
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

    # BYO mode: the job's inference.yaml is authoritative (num_classes,
    # image_size, top_k, cluster knobs). InferenceConfig only carries
    # submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        auto_select_workers=True,
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Image Classification"),
        inference=inference,
        from_files=str(ROOT),         # ship the job folder as-is, skip generation
    )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.api_key}")
    print()

    # First boot allocates the model on each replica (and downloads
    # torchvision's ImageNet weights when no checkpoint is shipped).
    print("[submit] waiting for deployment to become RUNNING…")
    job.wait_until_ready(timeout=1200, poll=10)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
