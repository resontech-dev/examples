# job_csgo — Object Detection (YOLOv5 + CS:GO)

Federated fine-tuning of a YOLOv5n object detector for CS:GO character detection.

## Model

- **Name**: `keremberke/yolov5n-csgo`
- **HF link**: https://huggingface.co/keremberke/yolov5n-csgo
- **Architecture**: YOLOv5n (nano) — 1.9 M parameters
- **Library**: [`yolov5`](https://pypi.org/project/yolov5/) pip package (v7.0.14)
- **Pretrained**: COCO → fine-tuned by `keremberke` on CS:GO data

## Dataset

- **Name**: `keremberke/csgo-object-detection`
- **HF link**: https://huggingface.co/datasets/keremberke/csgo-object-detection
- **Source**: Roboflow Universe — `asd-culfr/wlots`
- **Format**: COCO JSON → converted to YOLO TXT by `build_shards.py`
- **Train/Val**: 3,879 / 383 images
- **Classes** (4): `ct`, `cthead`, `t`, `thead`

## Validation

- ✅ Static checks pass
- ✅ Simulator: configs parse, server starts, round 0 dispatched, executor invokes `fl_train_model` (fails only at `import yolov5` because not installed in sim — expected)
- ✅ Platform: ran successfully end-to-end across 5 rounds with 4 workers (verified earlier in conversation)
- ✅ Inference: `test_fl_model.py` loaded `result.pt` and ran detection (`121/121 tensors loaded`)

## Training metrics (observed on platform)

| Metric | Value |
|---|---|
| Per-step time | ~0.5 s (RTX 4090-class) |
| Per-round wall time | ~30-45 s (4 workers parallel) |
| **Total FL time** | **~3-5 min** (5 rounds × 2 epochs) |
| Effective epochs | 10 (= 2 local × 5 rounds) |
| Per-shard size | 35.7 MB |
| Per-round transfer | 6.8 MB (full state-dict, both directions) |
| Per-shard images | 970 train + 383 val |

## Hardware requirements

- **Min GPU**: any modern card with 4 GB VRAM (T4, RTX 3060, etc.)
- **Per-worker disk**: ~150 MB (shard + cache)
- **HF download**: ~7 MB (yolov5n-csgo weights)

## Quick start

```bash
cd /home/pyaremenko/examples/job_csgo
# shards already built (35-36 MB × 4)
ls shards/  # shard_0.zip ... shard_3.zip
```

Upload structure:
```
jobs/job_csgo/
├── scripts/         (model_def, yolo_utils, custom_*)
├── configs/         (config_fed_*.json)
├── requirements/    (requirements.txt — yolov5, torch>=2.7.1)
└── shards/          (4 zips)
```

## Original training recipe (for FL ↔ centralized comparison)

The card publishes only the *re-fine-tune* invocation, not the original recipe. Defaults below are from yolov5's `hyp.scratch.yaml` (used when the card's example command is run):

| Param | Value | Source |
|---|---|---|
| Optimizer | SGD, momentum=0.937 | yolov5 default |
| LR (lr0) | **0.01** | yolov5 default |
| LR final (lrf) | 0.01 (= lr0 × 0.01) | yolov5 default |
| Schedule | linear/cosine + warmup 3 epochs | yolov5 default |
| Batch size | **16** | card example |
| Epochs | 10 | card example |
| Image size | 640 | card example |
| Weight decay | 0.0005 | yolov5 default |
| Augmentations | mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5 | yolov5 hyp.scratch |
| Loss | CIoU box + BCE obj/cls | yolov5 default |
| Final metric | mAP@0.5 = **0.908** on csgo val | https://huggingface.co/keremberke/yolov5n-csgo |

**FL config matches**: `learning_rate: 0.01`, `batch_size: 16`, `local_epochs: 2`, 5 rounds → effective 10 epochs ≈ centralized.

## Caveats

- yolov5 7.0.14 is incompatible with PyTorch 2.6+ default `weights_only=True` — handled by the `torch.load` shim in `yolo_utils.py`.
- yolov5 forces `torch.use_deterministic_algorithms(True)` which breaks CuBLAS — neutralized by the no-op patch.
- yolov5's `check_dataset` resolves `train:`/`val:` relative to its install dir if `path:` is missing — handled by `_runtime_data_yaml` injecting `path: <data_root>`.
