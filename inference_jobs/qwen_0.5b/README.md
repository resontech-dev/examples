# 08_qwen_0.5b_sdk

## Folder layout

```
.
├── inference.yaml          # REQUIRED — model + cluster config
├── scripts/                # REQUIRED — at least one .py (the predictor)
│   └── llm_serve.py
├── sdk_utils/              # submit + call the deployment from Python
│   ├── submit.py
│   └── predict.py
├── model/                  # optional — checkpoint files, referenced from inference.yaml
└── .env.example            # copy to .env, fill in credentials
```

`inference.yaml` points `module:` at your entry script and `class_name:` at
the predictor class.

You can deploy either from the SDK (`sdk_utils/submit.py`, below) or by pasting
`scripts/llm_serve.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit> — same result.

### Shipping these files with the SDK

`rt_submit_inference` takes explicit, keyword-only paths — `inference.yaml`
(one file) and `scripts/` (a directory, **copied to storage recursively**):

```python
inference_yaml="inference.yaml",   # REQUIRED — one file
scripts_dir="scripts",             # REQUIRED — one dir, copied recursively
# model_file="model/weights.pt",   # OPTIONAL — not used here (weights pull from HF Hub)
# sample_data_dir="sample_data",   # OPTIONAL — dashboard playground inputs
```

> **Migration from the old API.** `from_files=` / `scripts=` (the
> folder-convention BYO API, plus the flat-mode walker and exclusion tables)
> are removed. Replace a single `from_files="./bundle"` with the four explicit
> paths above — one extra line per file shipped, but the SDK no longer
> constrains your local folder names; point it at any layout.

## 1. Submit

```python
from resontech import InferenceConfig, ResonTech, ResonTechConfig

sdk = ResonTech(ResonTechConfig(
    base_url="https://api.beta.reson.tech",
    s3_endpoint="https://s3.beta.reson.tech",
    email="...", password="...",
    s3_access_key_id="...", s3_secret_access_key="...",
))
sdk.login()

job = sdk.rt_submit_inference(
    name="Qwen 0.5B Chat",
    inference_yaml="inference.yaml",      # REQUIRED — one file
    scripts_dir="scripts",                # REQUIRED — one dir, copied recursively
    inference=InferenceConfig(
        visibility="PRIVATE",             # or "PUBLIC"
        auto_select_workers=True,
        # only_mine_workers=True,         # restrict to your own GPU pool
    ),
)
job.wait_until_ready(timeout=1800)
print(job.endpoint, job.api_key)
```

Run (from this folder):
```bash
cp .env.example .env        # fill in your credentials, then:
python sdk_utils/submit.py  # prints RESON_INFERENCE_URL + API key when RUNNING
```

## 2. Predict

```python
from resontech import InferenceClient

client = InferenceClient(
    url="<RESON_INFERENCE_URL from submit>",
    api_key="<RESON_INFERENCE_API_KEY from submit>",
    timeout=600.0,                        # None = no client-side timeout
)

result = client.predict_json({
    "prompt": "Explain attention in one sentence.",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
})
print(result["text"])
```

Run (from this folder, after adding the printed values to `.env`):
```bash
python sdk_utils/predict.py "Explain attention in one sentence."
```
