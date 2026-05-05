# job_classify — Image Classification (ViT-base + Food-101)

Federated fine-tuning of a Vision Transformer on Food-101 (101 food classes, 75,750 train images at 224×224). A non-trivial classification task with a published reference model.

## Model

- **Backbone**: [`vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) (86 M parameters)
- **Library**: [`timm`](https://github.com/huggingface/pytorch-image-models)

## Reference model with published Food-101 metrics

[`nateraw/food`](https://huggingface.co/nateraw/food) — ViT-base fine-tuned on Food-101.

**Verbatim from the model card:**
| Metric | Value |
|---|---|
| Accuracy | **0.8913** |
| Validation Loss | **0.4501** |
| Final training loss | 0.0452 |
| Epochs | 5 |
| Batch size | 128 |
| Learning rate | 2e-4 |

This is the baseline your federated training extends. Goal: match or approach 0.89 accuracy via FL aggregation across 4 workers.

## Dataset

- **Name**: `ethz/food101`
- **HF link**: https://huggingface.co/datasets/ethz/food101
- **Train / Val**: 75,750 / 25,250 images, 224×224 RGB
- **Classes** (101): apple_pie, baby_back_ribs, baklava, ..., waffles
- **Per-shard size**: ~19,000 train + 25,250 val (val duplicated across shards)

## Validation

- ✅ Static checks pass
- ✅ `model_def` imports with `timm` installed
- ⚠️ Simulator end-to-end: not run (shards not built; build downloads ~5 GB)
- ⚠️ Platform: not yet run

## Training cost (your FL scenario)

| Metric | Value |
|---|---|
| Per-step time (batch=32, 224×224, ViT-base) | ~0.5 s on RTX 4090 |
| Per-round wall time | ~30-60 min (19k images / 32 batch × 2 epochs) |
| **Total FL time** | **~1-3 hours** (5 rounds × 4 workers parallel) |
| Per-round transfer | ~340 MB (ViT-base fp32 state-dict) |
| Expected acc. after 5 FL rounds | **0.85-0.89** (within reach of nateraw's 0.8913 baseline) |

## Hardware

- **Min GPU**: 8 GB VRAM (ViT-base fits at batch=32 in fp16)
- **Per-worker disk**: ~6 GB (Food-101 shard + ImageNet-21k pretrained ViT cache)

## Quick start

```bash
cd /home/pyaremenko/examples/job_classify
pip install datasets pillow
python build_shards_hf.py   # downloads Food-101 (~5 GB), builds 4 shards
```

To swap dataset: edit `HF_DATASET` in `build_shards_hf.py`. Any HF image-classification dataset with `image` + `label`/`fine_label` works.

## Caveats

- **Food-101 is hard**: visually similar classes (e.g. spaghetti carbonara vs. spaghetti bolognese, multiple soup types). nateraw needed 5 full epochs at batch=128 to hit 0.89; FL will probably hit 0.85 with the cheaper schedule.
- timm ViT models are loaded with `num_classes=N` argument at build time — head is reshaped to match dataset's class count, all 101 classes train end-to-end.
