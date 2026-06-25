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

`from_files=` ships this whole tree to S3 as-is. `inference.yaml` points
`module:` at your entry script and `class_name:` at the predictor class.

You can deploy either from the SDK (`sdk_utils/submit.py`, below) or by pasting
`scripts/llm_serve.py` + `inference.yaml` into the web wizard at
<https://beta.reson.tech/dashboard/inference/submit> — same result.

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
    from_files=".",                       # ship this folder as-is
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
