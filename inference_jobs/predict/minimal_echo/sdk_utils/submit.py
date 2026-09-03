"""
Deploy the minimal echo predictor to ResonTech via the SDK.

BYO mode — point the SDK at this job's ``inference.yaml`` (one file) and
``scripts/`` (a directory, copied to storage recursively). No weights, no
GPU — this just proves the predictor contract (auth + networking + routing)
end-to-end.

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


HERE = Path(__file__).resolve().parent     # .../minimal_echo/sdk_utils
ROOT = HERE.parent                          # .../minimal_echo  (holds inference.yaml + scripts/)
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

    # BYO mode: the job's inference.yaml is authoritative (class_name,
    # init_args, cluster knobs). InferenceConfig only carries submit-time knobs.
    inference = InferenceConfig(
        visibility=os.getenv("VISIBILITY", "PRIVATE"),
        auto_select_workers=True,
        only_mine_workers=True,
    )

    job = sdk.rt_submit_inference(
        name=os.getenv("JOB_NAME", "Minimal Echo"),
        inference=inference,
        inference_yaml=str(ROOT / "inference.yaml"),   # REQUIRED — one file
        scripts_dir=str(ROOT / "scripts"),             # REQUIRED — dir, copied recursively
        # no model_file / sample_data_dir — echo needs no weights.
    )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    # No weights to download — boot is fast.
    print("[submit] waiting for deployment to become RUNNING…")
    job.wait_until_ready(timeout=600, poll=5)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
