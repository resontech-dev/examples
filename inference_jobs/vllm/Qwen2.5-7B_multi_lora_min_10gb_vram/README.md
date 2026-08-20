# Qwen2.5-7B + N customer LoRA adapters (vLLM multi-LoRA)

The fine-tuned-business-model serving shape (`lora-qwen25-7b`, priority
v1, Apache-2.0): **one base model in VRAM, 8–16 customer adapters** loaded
on demand from S3 and selected per request via the `model` field. This is
the economics feature — 20 customer fine-tunes ≠ 20 GPUs; per-adapter
monthly billing hangs off the model id.

Shared vLLM contract & defaults: [../README.md](../README.md).

## Resources

| VRAM min→rec | CPU | RAM | Min free disk | Tier |
|---|---|---|---|---|
| 10 → 12 GB | 4 | 24 GB | ~28 GB (image 15 + 1.3×5.5 GB base; adapters are ~100–300 MB each) | 12gb |

Suggested GPUs: RTX 5070 / 4070 / 3060-12G (12 GB — as configured); any
bigger card raises the adapter count and context.

## How adapter selection works

```
model="qwen-7b"               → base model, no adapter
model="qwen-7b:acme-support"  → base + adapter folder "acme-support"
```

The engine resolves `<dynamic_lora_loading_path>/acme-support/` (a
standard PEFT checkpoint: `adapter_config.json` + adapter weights) on
first use and caches it. Set the S3 prefix in `serve_module.py` (or the
`LORA_S3_PREFIX` env var) before deploying.

## Deploy & test

```bash
cp .env.example .env
python submit.py
python predict.py                          # base model
python predict.py --adapter acme-support  # base + adapter
```

Or paste `scripts/serve_module.py` + `inference.yaml` into
<https://beta.reson.tech/dashboard/inference/submit>.

## What to change

| where (`serve_module.py`) | for what |
|---|---|
| `lora_config.dynamic_lora_loading_path` | **required** — your adapter bucket prefix |
| `max_lora_rank` | 32 shipped; must be ≥ the rank customers train (64 common) |
| `max_loras` / `max_num_adapters_per_replica` | resident-per-batch / cached-per-replica counts |
| `model_source` | swap base (Qwen3-8B-AWQ works identically); dense bases only |

## ⚠️ The safety note that matters for this product

Fine-tuning strips alignment — a small LoRA can remove refusals entirely,
and this serving stack will happily load it. Before hosting third-party
adapters publicly: AUP clause + an eval gate (see
`serve-coding-llms.md §8`). Adapter uploads are customer code — treat the
S3 prefix as a trust boundary.
