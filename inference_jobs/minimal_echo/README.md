# Minimal echo

The smallest possible predictor — it echoes back whatever bytes you POST.
No weights, no GPU, no dependencies. Deploy it first to validate auth,
networking, and request routing before you wire up a real model.

## Folder layout

```
.
├── inference.yaml          # REQUIRED — class to load + cluster config
├── scripts/                # REQUIRED — at least one .py (the predictor)
│   └── echo_serve.py       #   EchoPredictor.predict(data: bytes) -> dict
├── sdk_utils/              # submit + call the deployment from Python
│   ├── submit.py
│   └── predict.py
└── .env.example            # copy to .env, fill in credentials
```

`sdk_utils/submit.py` ships these to storage with explicit, keyword-only paths —
`inference.yaml` (one file) and `scripts/` (a directory, **copied recursively**):

```python
inference_yaml="inference.yaml",   # REQUIRED — one file
scripts_dir="scripts",             # REQUIRED — one dir, copied recursively
# model_file="model/weights.pt",   # OPTIONAL — not used here (echo has no weights)
# sample_data_dir="sample_data",   # OPTIONAL — dashboard playground inputs
```

> **Migration from the old API.** `from_files=` / `scripts=` (the
> folder-convention BYO API) are removed. Replace a single
> `from_files="./bundle"` with the explicit paths above — the SDK no longer
> constrains your local folder names; point it at any layout.

## Deploy

You can deploy this job two equivalent ways — from the SDK, or by pasting the
files into the web UI. Pick one.

### A. From the SDK

```bash
cp .env.example .env        # fill in RESON_EMAIL / PASSWORD / S3 keys
python sdk_utils/submit.py  # deploys, then prints RESON_INFERENCE_URL + API key
```

Copy the printed `RESON_INFERENCE_URL` and `RESON_INFERENCE_API_KEY` into `.env`, then:

```bash
python sdk_utils/predict.py "hello world"
# single → 'hello world'
#         echo: 'echo: hello world'  bytes_received=11 request_index=1 (0.04 ms)
```

### B. From the UI

Paste the contents of `scripts/echo_serve.py` and `inference.yaml` into the
submit wizard at <https://beta.reson.tech/dashboard/inference/submit>. The
endpoint and API key appear in the dashboard once the deployment is RUNNING.
Drop them into `.env` and call it with `python sdk_utils/predict.py`, or with curl:

```bash
curl -X POST "$RESON_INFERENCE_URL" \
  -H "X-API-Key: $RESON_INFERENCE_API_KEY" \
  -H "Content-Type: text/plain" \
  --data "hello world"
# -> {"echo":"echo: hello world","bytes_received":11,"request_index":1,"elapsed_ms":0.04}
```

## Extend it

Two minutes of editing turns this into your real predictor:

1. Rename `EchoPredictor` → `MyPredictor` in `scripts/echo_serve.py`, and
   update `class_name:` in `inference.yaml` to match.
2. Add a `load_weights(self, path)` method if you want the platform to
   mount a model file (set `model.path:` in `inference.yaml`).
3. Replace `predict()` with your inference logic. Return any JSON-serialisable dict.
