# Mistral Small 24B + N customer LoRA adapters (vLLM multi-LoRA)

The **EU-friendly mid-size LoRA base** (`lora-mistral-small-24b`,
Apache-2.0): regulated buyers get their own fine-tuned model — their tone,
vocabulary, format compliance — served as adapters on one shared AWQ base.
Same serving shape as the Qwen2.5-7B sibling, bigger brain, fewer adapters
per card.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 16 → 24 GB | 4 | 48 GB | ~39 GB (image 15 + 1.3×13 GB base; adapters ~200–600 MB each) | 24gb |

Suggested GPUs: RTX 3090 / 4090 / L4 / A10G (24 GB — as configured, ~8
cached adapters); 2×24 GB TP=2 → the 32B coder base for per-repo code
adapters (`lora-qwen25-coder-32b` shape).

## How adapter selection works

```
model="mistral-24b"               → base model
model="mistral-24b:acme-reports"  → base + PEFT checkpoint at
                                    <dynamic_lora_loading_path>/acme-reports/
```

Set the S3 prefix in `serve_module.py` (or `LORA_S3_PREFIX`) before
deploying. Adapters **must be trained against the same base revision** —
an adapter trained on the official bf16 base will load against this AWQ
re-upload but quality shifts; standardize the training base.

## Deploy & test

```bash
cp .env.example .env
python submit.py
python predict.py                           # base model
python predict.py --adapter acme-reports    # base + adapter
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `lora_config.dynamic_lora_loading_path` | **required** — your adapter bucket prefix |
| `model_source` | ⚠️ community AWQ — vet before selling |
| `max_lora_rank` / `max_loras` | match customer training rank; VRAM-bounded |

## ⚠️ Safety

Fine-tuning strips alignment — the AUP + eval-gate note from the Qwen
LoRA sibling applies verbatim. Treat the adapter S3 prefix as a trust
boundary.
