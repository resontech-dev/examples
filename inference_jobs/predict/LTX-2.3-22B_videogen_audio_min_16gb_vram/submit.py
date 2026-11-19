"""
Deploy the LTX-2.3 (video + audio) predictor to ResonTech.

Ships this folder's ``inference.yaml`` and ``scripts/`` (recursively). No
weights upload: the predictor pulls ~72 GB from HF Hub in __init__.

The Gemma 3 text encoder is gated. Put a read token of an account that
accepted the Gemma terms in .env as HF_TOKEN: this script copies
inference.yaml to a temp file with ``init_args.hf_token`` filled in and
submits that copy, so the token never lands in the repo. It does land in
the job's config on the platform: use a fine-grained read-only token
limited to google/gemma-3-12b-it-qat-q4_0-unquantized.

Run
---
    cp .env.example .env  # RESON_API_KEY (rsk_…) + S3 keys + HF_TOKEN
    python submit.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceConfig, ResonTech, ResonTechConfig


HERE = Path(__file__).resolve().parent
os.chdir(HERE)
load_dotenv(HERE / ".env")

TOKEN_LINE = "    hf_token: null"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[submit] Missing required env var: {name}")
    return value


def _yaml_with_token(tmp: Path) -> Path:
    """inference.yaml with the HF token filled in, written outside the repo."""
    token = _required("HF_TOKEN")
    lines = (HERE / "inference.yaml").read_text().splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(TOKEN_LINE)]
    if len(hits) != 1:
        sys.exit(f"[submit] expected one line starting with {TOKEN_LINE.strip()!r} in inference.yaml")
    lines[hits[0]] = f"    hf_token: {json.dumps(token)}\n"   # JSON string = valid YAML scalar
    path = tmp / "inference.yaml"
    path.write_text("".join(lines))
    return path


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
        # Ordered GPU preference; FP8 needs Ada / Hopper / Blackwell (no 3090,
        # no A100). 48 GB+ runs resident; smaller cards stream layers from RAM.
        gpu_models=["l40s", "h100_80gb", "rtx_5090", "rtx_4090", "rtx_5080", "rtx_5070ti", "rtx_4080"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "2.50")),
        only_mine_workers=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        job = sdk.rt_submit_inference(
            name=os.getenv("JOB_NAME", "LTX-2.3-22B-video-audio"),
            inference=inference,
            inference_yaml=str(_yaml_with_token(Path(tmp))),
            scripts_dir=str(HERE / "scripts"),
        )

    print()
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    print("[submit] first boot pulls ~72 GB from HF Hub: expect a long wait")
    print("[submit] waiting for deployment to become RUNNING…")
    job.wait_until_ready(timeout=5400, poll=15)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
