# job_yolo11 — Object Detection (YOLO11m + COCO 2017)

Federated training of **YOLO11m** on **COCO 2017** — the canonical benchmark Ultralytics actually publishes mAP numbers against. Real-scale detection task (118k train images, 80 classes), unlike the 4-class CS:GO niche.

## Model

- **Name**: `Ultralytics/YOLO11`
- **HF link**: https://huggingface.co/Ultralytics/YOLO11
- **Variant used**: `yolo11m.pt` (medium, 20.1 M parameters)
- **Library**: [`ultralytics`](https://github.com/ultralytics/ultralytics)

## Published metrics (verbatim from `Ultralytics/YOLO11` HF card)

COCO val2017 mAP@50-95:

| Variant | Params | mAP@50-95 |
|---|---|---|
| YOLO11n | 2.6 M | 39.5 |
| YOLO11s | 9.4 M | 47.0 |
| **YOLO11m** | **20.1 M** | **51.5** |
| YOLO11l | 25.3 M | 53.4 |
| YOLO11x | 56.9 M | 54.7 |

For comparison, YOLOv8m (25.9 M params) hits 50.2 — YOLO11m beats it with fewer parameters.

Reference: https://huggingface.co/Ultralytics/YOLO11 (Performance section).

## Dataset

- **Name**: `rafaelpadilla/coco2017`
- **HF link**: https://huggingface.co/datasets/rafaelpadilla/coco2017
- **Origin**: Microsoft COCO 2017, https://cocodataset.org/
- **Examples**: ~118,000 train / ~5,000 val
- **Classes** (80): person, bicycle, car, motorcycle, ..., toothbrush
- **Format**: COCO JSON → converted to YOLO TXT by `build_shards_hf.py`

## Validation

- ✅ Static checks pass
- ✅ Adapter API: `make_fl_adapter` returns `(yolo_model, fl_train_model)`
- ⚠️ Simulator end-to-end: not run (ultralytics not in test env)
- ⚠️ Platform: not yet run

## Training cost (your FL scenario)

| Metric | Value |
|---|---|
| Per-step time (batch=16, 640×640, YOLO11m) | ~0.4 s on RTX 4090 |
| Per-round wall time | ~1-2 hours (29.5k images / 16 batch × 2 epochs) |
| **Total FL time** | **~6-10 hours** (5 rounds × 4 workers parallel) |
| Per-round transfer | ~80 MB (YOLO11m fp32 state-dict) |
| Expected mAP@50-95 after FL | **48-52** (within 2-3 pts of centralized 51.5 baseline) |

## Hardware

- **Min GPU**: 12 GB VRAM (YOLO11m + 640×640 batch=16, fp16)
- **Recommended**: 24 GB for batch=32+
- **Per-worker disk**: ~5 GB (COCO shard ~4-5 GB + yolo11m.pt cache ~40 MB)

## Quick start

```bash
cd training/nvflare/job_yolo11
pip install datasets pillow ultralytics
python build_shards_hf.py   # downloads COCO 2017 (~25 GB), builds 4 shards (~5-6 GB each)
```

For a smoke test instead of full COCO, set `MAX_EXAMPLES = 4000` in `build_shards_hf.py` (~1k images per shard, ~10 min training per round).

## Why YOLO11m (not n or x)

- YOLO11n (39.5 mAP) — too weak for COCO; FL won't recover the gap
- **YOLO11m (51.5 mAP)** ← sweet spot for FL: trainable on 12 GB GPUs, strong starting point
- YOLO11l/x (53.4 / 54.7 mAP) — diminishing returns, demands 24 GB GPUs

## Original training recipe (for FL ↔ centralized comparison)

From Ultralytics' [default cfg](https://docs.ultralytics.com/usage/cfg/) — the same defaults `model.train(data='coco.yaml')` uses:

| Param | Value | Source |
|---|---|---|
| Optimizer | SGD (auto-selected for YOLO11m) | Ultralytics auto |
| LR (lr0) | **0.01** | default |
| LR final (lrf) | lr0 × 0.01 = 1e-4 | default |
| Schedule | linear decay | default |
| Warmup | 3.0 epochs (warmup_momentum=0.8, warmup_bias_lr=0.1) | default |
| Batch size | **16** per-device (Ultralytics' COCO uses effective ~64 across 8 GPUs) | default |
| Epochs | 100 (default); the released checkpoint used ~500 epochs with close-mosaic last 10 | default / repo README |
| Momentum | 0.937 | default |
| Weight decay | 0.0005 | default |
| Loss weights | box=7.5, cls=0.5, dfl=1.5 | default |
| Augmentations | hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, fliplr=0.5, mosaic=1.0 | hyp.scratch |
| Mixed precision | amp=True (fp16) | default |
| Final metric | **mAP@50-95 = 51.5** on COCO val2017 (20.1M params) | https://huggingface.co/Ultralytics/YOLO11 |

**FL config matches**: `learning_rate: 0.01`, `batch_size: 16`. To approximate the 100-epoch centralized run with 5 FL rounds × 4 workers, use `local_epochs: 20` (= 100/5). For a faster comparison, drop to `local_epochs: 5` and `num_rounds: 5` for a 25-epoch effective.

## Caveats

- COCO is **massive** (~25 GB). Initial dataset download will take time. Subsample with `MAX_EXAMPLES` for development.
- YOLO11 head is auto-rebuilt to match dataset's class count via `model.train(data=...)`. Round 0 starts with COCO 80-class head; if you switched dataset to non-80-class, head is reinitialized each round (same head-doesn't-accumulate caveat as job_csgo).
- Ultralytics requires `path:` absolute in `data.yaml` — `_runtime_data_yaml` handles it.
