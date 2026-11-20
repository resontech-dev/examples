"""
Deploy the Nemotron 3.5 ASR Streaming speech-to-text predictor to ResonTech.

Ships this folder's ``inference.yaml`` (one file) and ``scripts/``
(recursively) via explicit keyword paths. No weights upload: NeMo pulls the
checkpoints from HF Hub in __init__ (NeMo itself installs from PyPI).

Run
---
    cp .env.example .env  # fill in RESON_API_KEY (rsk_…) + S3 keys
    python submit.py                       # Nemotron only (init_args.final_model as in the yaml)
    python submit.py --final parakeet      # two passes: Nemotron draft + Parakeet v3 final text
    python submit.py --final whisper       # ... + Whisper large-v3
    python submit.py --final whisper-turbo # ... + Whisper large-v3-turbo

``--final`` submits a temporary copy of inference.yaml with ``final_model``
set, so several variants can run side by side from one folder.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from resontech import InferenceConfig, ResonTech, ResonTechConfig


HERE = Path(__file__).resolve().parent
os.chdir(HERE)
load_dotenv(HERE / ".env")

FINAL_LINE = "    final_model:"
FINAL_CHOICES = ("none", "parakeet", "whisper", "whisper-turbo")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[submit] Missing required env var: {name}")
    return value


def _yaml_with_final(tmp: Path, final: str) -> Path:
    """inference.yaml with init_args.final_model set, written outside the repo."""
    lines = (HERE / "inference.yaml").read_text().splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(FINAL_LINE)]
    if len(hits) != 1:
        sys.exit(f"[submit] expected one line starting with {FINAL_LINE.strip()!r} in inference.yaml")
    value = "null" if final == "none" else f'"{final}"'
    lines[hits[0]] = f"{FINAL_LINE} {value}\n"
    path = tmp / "inference.yaml"
    path.write_text("".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", choices=FINAL_CHOICES, help="second-pass model (default: as in inference.yaml)")
    args = ap.parse_args()

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
        # Ordered GPU preference (>= 8 GB VRAM per card); first entry is filled first.
        gpu_models=["rtx_5070ti", "rtx_5070", "rtx_4070", "rtx_2080ti", "rtx_4080", "rtx_5080", "rtx_4090", "rtx_3090"],
        max_price_per_gpu_hour=float(os.getenv("MAX_PRICE_PER_GPU_HOUR", "2.50")),
        only_mine_workers=True,
    )

    name = os.getenv("JOB_NAME") or "Nemotron 3.5 ASR Streaming" + (
        f" + {args.final}" if args.final and args.final != "none" else ""
    )
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = _yaml_with_final(Path(tmp), args.final) if args.final else HERE / "inference.yaml"
        job = sdk.rt_submit_inference(
            name=name,
            inference=inference,
            inference_yaml=str(yaml_path),
            scripts_dir=str(HERE / "scripts"),
        )

    print()
    print(f"[submit] {name}")
    print("[submit] Save these two values for predict.py:")
    print(f"    RESON_INFERENCE_URL = {job.endpoint}")
    print(f"    RESON_INFERENCE_API_KEY = {job.predict_api_key}")
    print()

    print("[submit] first boot installs NeMo from PyPI and pulls the checkpoints: expect several minutes on a cold host")
    print("[submit] waiting for deployment to become RUNNING…")
    job.wait_until_ready(timeout=3600, poll=15)
    print(f"[submit] ✅ {job.id} is RUNNING")


if __name__ == "__main__":
    main()
