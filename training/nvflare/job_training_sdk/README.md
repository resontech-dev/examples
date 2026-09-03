# job_training_sdk

Minimal, runnable federated-learning job built on the ResonTech Python SDK.

A tiny CNN on MNIST, submitted to the ResonTech platform with one Python call.
Use it as a clone-and-edit base for your own model + dataset.

## Layout

```
job_training_sdk/
├── model_def.py        # Your model + fl_train(...) + fl_validate(...)
├── build_shards.py     # Builds shards/shard_N.zip — one per worker
├── submit.py           # Reads .env, logs in, calls sdk.rt_submit(...)
├── requirements.txt    # Installed on every worker before training starts
├── .env.example        # Credential placeholders
└── README.md
```

You only edit two files day-to-day: `model_def.py` (what to train) and
`submit.py` (how to schedule it).

## Run it

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Credentials

```bash
cp .env.example .env
# Edit .env — fill in RESON_API_KEY, RESON_S3_KEY, RESON_S3_SECRET.
# Api key: dashboard → /profile → Developer; S3 keys: Profile → Storage.
```

### 3. Build shards

One zip per worker. The script downloads MNIST the first time (~12 MB) and
stratifies it across `--num-shards` workers.

```bash
python build_shards.py --num-shards 1
```

Rebuild with a different `--num-shards` whenever you change worker count.
Old shard zips are auto-pruned from S3 by the SDK on the next submit.

### 4. Submit

```bash
python submit.py
```

Output:
```
[submit] logged in as you@example.com
[submit] submitting…
[submit] submitted: clx1234abcdef0000
[submit] dashboard: https://beta.reson.tech/dashboard/jobs/clx1234abcdef0000
```

Open the dashboard link to watch rounds tick by.

## Tweaking

All knobs are env vars read by `submit.py` — no code edits needed for the
common ones:

| Var | Default | What |
|---|---|---|
| `NUM_CLIENTS` | `1` | Workers per round. Must match the shard count. |
| `NUM_ROUNDS` | `3` | Total FL rounds. |
| `LOCAL_EPOCHS` | `1` | Epochs each worker runs per round. |
| `BATCH_SIZE` | `64` | DataLoader batch size on the worker. |
| `LR` | `1e-3` | Learning rate. |
| `JOB_NAME` | `example-sdk` | Display name in the dashboard. |
| `QUANTIZATION` | unset | `float16`, `bfloat16`, `blockwise8`, `float4`, `normfloat4`, or `adaquant` — halves+ the per-round transfer. See [`sdk-training-hyperparameters`](https://reson.tech/docs/sdk-training-hyperparameters). |

Example:
```bash
NUM_CLIENTS=3 NUM_ROUNDS=5 QUANTIZATION=float16 python submit.py
```

(Don't forget to rebuild shards after bumping `NUM_CLIENTS`.)

## Adapting to your own model / data

**Swap the model.** Replace the `Model` class in
[`model_def.py`](model_def.py). Anything you pass to its constructor goes
into `MODEL_ARGS` in [`submit.py`](submit.py) — the SDK forwards it
verbatim to every site.

**Swap the dataset.** Two places:

1. [`build_shards.py`](build_shards.py): replace `_load_dataset(...)` with
   whatever produces your `(x_train, y_train, x_val, y_val)` tensors (or
   write totally different files into each shard staging dir — the SDK only
   cares about the resulting `shard_*.zip` shape).
2. [`model_def.py`](model_def.py): replace `_load_split(data_root, split)`
   to match. `data_root` is the unpacked-shard directory on the worker.

**Add training knobs.** Put non-standard hyperparameters in `TRAIN_EXTRA`
in [`submit.py`](submit.py); they land in the `env` dict your
`fl_train(model, env, out_dir, logger)` receives. The fixed `TrainingConfig`
fields (`local_epochs`, `batch_size`, `learning_rate`) become
`env["EPOCHS"] / ["BATCH_SIZE"] / ["LR"]` per SDK convention.

**Add a custom executor or persistor.** Pass `executor=YourExecutor` /
`persistor=YourPersistor` to `sdk.rt_submit(...)`. See
[`sdk-custom-classes`](https://reson.tech/docs/sdk-custom-classes).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No workers online` | Check the cluster dashboard. If you only have your own node, start the worker daemon there first. |
| `shard count ≠ worker count` | Rerun `build_shards.py --num-shards <N>` to match `NUM_CLIENTS`. |
| Job sits in PENDING > 1 min | Provisioning the FL server takes a minute on cold start — wait. The SDK's default HTTP timeout (600 s) covers it. |
| `ModuleNotFoundError` on the worker | Add the missing package to `requirements.txt`; it ships to every worker. |

## What this example doesn't show

Kept out on purpose:

- Local-only fallback path (this example only submits remotely).
- Resumable checkpoints / mid-training pause.
- Worker GPU pinning, custom NVFlare workflows, DP/HE filters.

For those, see [`sdk-configuration`](https://reson.tech/docs/sdk-configuration)
(advanced fields on `FederationConfig` including `extra_client_filters` /
`extra_server_filters`) and the full sample at
[`samlora_resontech_package/`](../../resontech-sdk/samlora_resontech_package).
