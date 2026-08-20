# Image classification

Image in → ranked class labels out. Default architecture is ResNet50;
swap it inside `scripts/classifier_serve.py:__init__` for whatever you trained.
With no checkpoint in `model/`, it falls back to ImageNet-pretrained ResNet50
so you can smoke-test the deploy before training.

## Folder layout

```
.
├── inference.yaml              # REQUIRED — model + cluster config
├── scripts/                    # REQUIRED — at least one .py (the predictor)
│   └── classifier_serve.py     #   ImageClassifier.predict(data: bytes) -> dict
├── model/                      # optional — drop classifier.pt here (git-ignored)
├── sdk_utils/                  # submit + call the deployment from Python
│   ├── submit.py
│   └── predict.py
└── .env.example                # copy to .env, fill in credentials
```

`sdk_utils/submit.py` ships these to storage with explicit, keyword-only paths —
`inference.yaml` (one file) and `scripts/` (a directory, **copied recursively**):

```python
inference_yaml="inference.yaml",                 # REQUIRED — one file
scripts_dir="scripts",                           # REQUIRED — one dir, copied recursively
# model_file="model/classifier.pt",             # OPTIONAL — uncomment to ship a trained checkpoint
# sample_data_dir="sample_data",                 # OPTIONAL — dashboard playground inputs
```

With no `model_file`, the deploy falls back to ImageNet-pretrained ResNet50.

> **Migration from the old API.** `from_files=` / `scripts=` (the
> folder-convention BYO API) are removed. Replace a single
> `from_files="./bundle"` with the explicit paths above — the SDK no longer
> constrains your local folder names; point it at any layout.

## Deploy

You can deploy this job two equivalent ways — from the SDK, or by pasting the
files into the web UI. Pick one.

### A. From the SDK

```bash
cp .env.example .env        # fill in RESON_API_KEY + S3 keys
python sdk_utils/submit.py  # deploys, then prints RESON_INFERENCE_URL + predict api key
```

Copy the printed `RESON_INFERENCE_URL` and `RESON_INFERENCE_API_KEY` into `.env`, then:

```bash
python sdk_utils/predict.py                       # classifies a sample image (URL)
python sdk_utils/predict.py ./cat.jpg             # a local file
python sdk_utils/predict.py https://…/dog.jpg     # a remote URL
```

### B. From the UI

Paste the contents of `scripts/classifier_serve.py` and `inference.yaml` into
the submit wizard at <https://beta.reson.tech/dashboard/inference/submit>. Grab
the endpoint and API key from the dashboard once the deployment is RUNNING, drop
them into `.env`, and call it with `python sdk_utils/predict.py`.

## What to change

| where                                                     | for what                                                  |
|-----------------------------------------------------------|-----------------------------------------------------------|
| `inference.yaml` → `init_args.num_classes`                | match your training head                                  |
| `inference.yaml` → `init_args.class_names`                | optional `[name, ...]` list, length must equal num_classes|
| `inference.yaml` → `init_args.image_size`                 | match your training preprocessing                         |
| `scripts/classifier_serve.py:__init__` → `resnet50(...)`  | swap for `efficientnet_b0`, `vit_b_16`, custom nn.Module… |
| `scripts/classifier_serve.py:__init__` → `self.transform` | match your training preprocessing exactly                 |
| `model/classifier.pt`                                     | drop your trained checkpoint here to skip the ImageNet fallback |
