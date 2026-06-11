"""
Submit a federated training job to ResonTech.

Run this from the directory containing it:

    python submit.py

It logs into the ResonTech API, uploads the rendered model/executor/configs
plus the shard zips, and starts a federated run on the platform.

To extend:
  * Edit ``MODEL_ARGS``    — kwargs forwarded to ``Model.__init__``.
  * Edit ``TRAIN_EXTRA``   — anything ``fl_train`` reads from ``env``.
  * Edit ``FederationConfig(...)`` — round count, worker count, quantization.
  * Point at a different model class by replacing the ``Model`` import below.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from resontech import (
    FederationConfig,
    ModelConfig,
    ResonTech,
    ResonTechConfig,
    TrainingConfig,
)

# Importing from model_def.py here also makes the file's source the
# ``model_def.py`` shipped to every worker — the SDK extracts it verbatim.
from model_def import Model


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")


# ── What the worker constructs the model with ────────────────────────────────
MODEL_ARGS: dict = {
    "num_classes": 10,
    "dropout": 0.1,
}


# ── Free-form hyperparameters surfaced to fl_train via ``env`` ───────────────
# (Stuff that wouldn't fit in TrainingConfig's typed fields lives here.)
TRAIN_EXTRA: dict = {
    "weight_decay": 1e-4,
    "num_workers": 2,
}


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"[submit] Missing required env var: {name} (see .env.example)")
    return value


def main() -> None:
    # ── Build & login ────────────────────────────────────────────────────────
    sdk = ResonTech(
        ResonTechConfig(
            base_url=os.getenv("RESON_BASE_URL", "https://api.beta.reson.tech"),
            s3_endpoint=os.getenv("RESON_S3_ENDPOINT", "https://s3.beta.reson.tech"),
            email=_required_env("RESON_EMAIL"),
            password=_required_env("RESON_PASSWORD"),
            s3_access_key_id=_required_env("RESON_S3_KEY"),
            s3_secret_access_key=_required_env("RESON_S3_SECRET"),
        )
    )
    user = sdk.login()
    print(f"[submit] logged in as {user.email}")

    # ── Decide round size against the live cluster ───────────────────────────
    stats = sdk.dashboard.worker_stats()
    requested = int(os.getenv("NUM_CLIENTS", "1"))
    clients = min(requested, stats.online_count)
    if clients < 1:
        sys.exit("[submit] No workers online — connect at least one and retry.")
    if clients != requested:
        print(f"[submit] only {clients}/{requested} worker(s) available — submitting for {clients}")

    # ── Verify shards exist and match the worker count ───────────────────────
    shards_dir = PROJECT_ROOT / "shards"
    shard_count = len(sorted(shards_dir.glob("shard_*.zip")))
    if shard_count != clients:
        sys.exit(
            f"[submit] shard count ({shard_count}) ≠ worker count ({clients}). "
            f"Rebuild: python build_shards.py --num-shards {clients}"
        )

    # ── SDK configs ──────────────────────────────────────────────────────────
    training = TrainingConfig(
        local_epochs=int(os.getenv("LOCAL_EPOCHS", "1")),
        batch_size=int(os.getenv("BATCH_SIZE", "64")),
        learning_rate=float(os.getenv("LR", "1e-3")),
        extra=TRAIN_EXTRA,
    )
    federation = FederationConfig(
        num_rounds=int(os.getenv("NUM_ROUNDS", "3")),
        min_clients=clients,
        job_name=os.getenv("JOB_NAME", "example-sdk"),
        # Opt-in bandwidth compression: halves the per-round transfer.
        # Drop / replace with "bfloat16" / "blockwise8" as needed — see
        # https://reson.tech/docs/sdk-training-hyperparameters
        quantization=os.getenv("QUANTIZATION") or None,
    )
    model_config = ModelConfig(model_args=MODEL_ARGS)

    # ── Submit ───────────────────────────────────────────────────────────────
    print("[submit] submitting…")
    job = sdk.rt_submit(
        model=Model,
        name=federation.job_name,
        shards_dir=str(shards_dir),
        requirements_txt=str(PROJECT_ROOT / "requirements.txt"),
        training=training,
        federation=federation,
        model_config=model_config,
    )
    print(f"[submit] submitted: {job.id}")
    print(f"[submit] dashboard: {job.dashboard_url}")


if __name__ == "__main__":
    main()
