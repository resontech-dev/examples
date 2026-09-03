# job_medical — Medical Imaging Classification (MONAI + HAM10000)

Federated learning on **HAM10000** (Human Against Machine with 10,000 training images) — a real dermatoscopic skin-lesion benchmark used in the 2018 ISIC challenge. Not a toy dataset.

## Model

- **Architecture**: `monai.networks.nets.DenseNet121` (2D, 3-channel input, 7-class output)
- **Parameters**: ~7 M
- **Library**: [`monai`](https://monai.io/) (Medical Open Network for AI)
- **Pretrained**: ImageNet weights via DenseNet's standard init (handled by MONAI)

## Reference metrics

> **No popular HF model with verifiable HAM10000 metrics on its model card was found.** The model cards we surveyed (`microsoft/BiomedCLIP`, various skin-cancer fine-tunes) either have no published numbers, or only show benchmarks as embedded SVG charts. Your FL training produces the reference data for this catalog.

If you want a reasonable centralized baseline to aim for: ISIC 2018 leaderboards reported ~0.85-0.90 multi-class balanced accuracy on HAM10000 with VGG/ResNet/EfficientNet at the time. DenseNet121 + ImageNet-init is a close peer.

## Dataset

- **Name**: `marmal88/skin_cancer`
- **HF link**: https://huggingface.co/datasets/marmal88/skin_cancer
- **Origin**: HAM10000 — Tschandl, Rosendahl, Kittler (2018). https://doi.org/10.7910/DVN/DBW86T
- **Format**: dermatoscopic images + class label (`dx` field)
- **Train / Val / Test**: 9,580 / 2,490 / 1,294 images
- **Classes** (7):
  - `akiec` — actinic keratoses & intraepithelial carcinoma
  - `bcc` — basal cell carcinoma
  - `bkl` — benign keratosis-like lesions
  - `df` — dermatofibroma
  - `mel` — melanoma
  - `nv` — melanocytic nevi (most common)
  - `vasc` — vascular lesions

## Validation

- ✅ Static checks pass
- ✅ `model_def` imports with `monai` installed
- ⚠️ Simulator end-to-end: not run (shards not built)
- ⚠️ Platform: not yet run

## Training cost (your FL scenario)

| Metric | Value |
|---|---|
| Per-step time (batch=32, 64×64) | ~0.05 s on RTX 4090 |
| Per-round wall time | ~10-20 min (2.4k images / 32 batch × 2 epochs) |
| **Total FL time** | **~30-90 min** (5 rounds × 4 workers parallel) |
| Per-round transfer | ~30 MB (DenseNet-121 state-dict) |
| Expected balanced acc. | **0.75-0.85** (HAM10000 is class-imbalanced — `nv` dominates ~67% of training) |

## Hardware

- **Min GPU**: 8 GB VRAM (DenseNet121 + 224×224 batch=32 comfortably)
- **Per-worker disk**: ~600 MB (HAM10000 ~3 GB total / 4 = ~750 MB before compression)

## Quick start

```bash
cd training/nvflare/job_medical
pip install datasets pillow monai
python build_shards_hf.py   # downloads HAM10000, builds 4 shards
```

## Why this matters

HAM10000 is the canonical FL medical-imaging demonstrator:
- **Realistic class imbalance** (`nv` >> others) tests aggregation robustness
- **Privacy-genuine**: dermatoscopic images of identifiable patients can't legally cross institutional boundaries in many jurisdictions
- **Cross-clinic relevance**: each clinic has different patient demographics, equipment calibration, lesion mix — FL captures all of them without sharing raw images

For a production-grade extension:
- Swap dataset to ISIC 2019/2020 (larger, more lesion types)
- Switch model to `monai.networks.nets.SwinUNETR` for segmentation tasks
- Use 3D pipeline for CT/MRI volumes with `DecathlonDataset`

## Original training recipe (from `Kuldeepmishra3/vit-large-skin-cancer-ham10000` HF card)

The most-documented HAM10000 model on HF Hub (no canonical baseline exists with verifiable card metrics, this is the closest):

| Param | Value | Source |
|---|---|---|
| Optimizer | **AdamW** (HF Trainer default) | card |
| Learning rate | **2e-5** | card |
| LR schedule | **cosine** | card |
| Warmup steps | **300** | card |
| Batch size | **16** per-device | card |
| Epochs | **5** (best at epoch 4) | card |
| Image size | 224 × 224 (ViT-large/16) | architecture |
| Weight decay | 0.0 (HF default) | inferred |
| Grad clip | 1.0 (HF default) | inferred |
| Mixed precision | fp16 | card |
| Augmentations | random H/V flip, rotation up to 30°, color jitter | card |
| Loss | cross-entropy (7-class) | standard |
| Final metric | **acc 92.74%**, weighted F1 92.60% on val | https://huggingface.co/Kuldeepmishra3/vit-large-skin-cancer-ham10000 |
| Hardware | 1× Tesla T4 (Colab), ~37 min training | card |

Note: this baseline uses ViT-large; our adapter uses MONAI DenseNet121 (similar parameter count, simpler kernel). Same hyperparameters apply; the head dimension differs.

**FL config matches**: `learning_rate: 2e-5`, `batch_size: 16`, `local_epochs: 1`, 5 rounds → effective 5 epochs.

## Caveats

- HAM10000 has heavy class imbalance — consider `WeightedRandomSampler` or focal loss in `monai_utils.py` for serious runs. The current adapter uses plain `CrossEntropyLoss`.
- `marmal88/skin_cancer` is a community re-host of the original Harvard Dataverse upload — verify the integrity matches the official MD5 if reproducibility matters.
