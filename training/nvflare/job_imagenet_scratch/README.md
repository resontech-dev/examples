# job_imagenet_scratch — ResNet-50 from scratch on ImageNet-1k (timm A2)

**Corporate-grade from-scratch training**: random-init ResNet-50, full ImageNet-1k, the canonical timm A2 recipe (`ResNet Strikes Back`, ICLR 2022). Tests federated aggregation under a real production training load — no pretrained shortcut.

## Model

- **Architecture**: ResNet-50 (25.6 M parameters), random init
- **Library**: [`timm`](https://github.com/huggingface/pytorch-image-models)
- **Reference checkpoint**: [`timm/resnet50.a2_in1k`](https://huggingface.co/timm/resnet50.a2_in1k) — what we're trying to reproduce via FL

## Reference recipe (verbatim from timm A2)

Source: [arXiv 2110.00476](https://arxiv.org/abs/2110.00476) "ResNet Strikes Back" + [timm/resnet50.a2_in1k](https://huggingface.co/timm/resnet50.a2_in1k)

| Param | Value |
|---|---|
| Optimizer | **LAMB**, momentum=0.9 |
| Learning rate | **5e-3** (peak) |
| LR schedule | **cosine** + 5-epoch warmup |
| Batch size | **2048** (across GPUs) |
| Epochs | **300** (A2; A1 = 600 epochs) |
| Weight decay | **0.02** |
| Loss | **BCE** + label smoothing 0.1 (replaces CE) |
| Augmentations | RandAugment(2, 7) + Mixup α=0.1 + CutMix α=1.0 (switch 0.5) + repeated augmentation |
| Stochastic depth | **0.05** |
| Mixed precision | fp16 AMP |
| Hardware | 1 node × 4 × V100-32GB, ~4.6 days for A1 (600 ep) |
| **Final metric** | **79.8% top-1, 95.1% top-5** on ImageNet-1k val | (A2)

## Dataset

- **Name**: `ILSVRC/imagenet-1k`
- **HF link**: https://huggingface.co/datasets/ILSVRC/imagenet-1k
- **Size**: 1,281,167 train + 50,000 val, 1000 classes
- **Per-shard**: ~320k train + 50k val (val duplicated across shards)
- **Disk**: ~38 GB per shard zip, ~150 GB total
- **Auth**: `huggingface-cli login` required (gated)

For development, set `MAX_EXAMPLES=40000` in `build_shards_hf.py` (~10k per shard, ~40-50 min/round on H100). Or swap `HF_DATASET = "frgfm/imagenette"` for a 10-class smoke-test (~5 min/round).

## FL config (matches A2 recipe across 4 workers × 5 rounds)

| FL knob | Value | Maps to centralized recipe |
|---|---|---|
| `learning_rate` | **5e-3** | LAMB lr |
| `batch_size` | **256** per worker (effective 1024 across 4 workers; close to A2's 2048) | A2 batch |
| `local_epochs` | **60** | 60 × 5 rounds = **300** epochs (matches A2) |
| `num_rounds` (server) | 5 | n/a |

## Compute estimate (4 workers × 1× H100 each, in parallel)

| Phase | Per worker | Wall time |
|---|---|---|
| Per-step time (batch 256, fp16) | ~0.4 s | — |
| Steps per local epoch | ~1,250 (320k / 256) | — |
| 60 local epochs | 75,000 steps × 0.4s = **8.3 hr** | — |
| **5 rounds × 8.3h + aggregation** | — | **~42-50 hr (≈2 days)** |

For comparison, single-worker centralized (1 H100, 300 epochs, 1.28M imgs):
- Per epoch: 1.28M / 256 batch × 0.4s = ~33 min
- 300 epochs = **~165 hr (≈7 days)**

**FL speedup: ~3.5×** (4 workers in parallel + aggregation overhead).

## Hardware

- **Min GPU per worker**: H100-80GB or A100-80GB at full 320k-shard scale
- **Workable on RTX 4090 (24 GB)** if `MAX_EXAMPLES` is reduced and `batch_size` dropped to 128

## Quick start

```bash
cd training/nvflare/job_imagenet_scratch
pip install datasets pillow timm
huggingface-cli login                    # for ILSVRC/imagenet-1k (gated)
python build_shards_hf.py                # downloads ~150 GB; takes hours
```

## Federation value proposition

This is the canonical demonstrator: 4 organizations each have their own ImageNet-class image archive (e.g. retailers with proprietary product imagery, hospitals with anonymized medical scans, manufacturers with defect photos). They want to jointly train a shared backbone but cannot pool images. FL gives each org:

- Trained model with quality close to centralized (within ~2-3% top-1)
- ~3.5× faster wall time than any one org alone
- No raw-image sharing
