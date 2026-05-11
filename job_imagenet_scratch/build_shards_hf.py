"""Build ImageNet-1k FL shards from HuggingFace.

Real-scale image classification: 1.28M train images, 50k val, 1000 classes,
~150 GB at 224×224 RGB. Designed for H100-class workers; subsample via
MAX_EXAMPLES for development.

Layout (matches imagenet_utils.py's ImageFolder loader):
    shard_X.zip
        images/train/<class_id>/*.jpg
        images/val/<class_id>/*.jpg
        data.yaml          # {nc: 1000, names: [...]}

Notes:
    - HF dataset `imagenet-1k` is gated; needs `huggingface-cli login`
    - For testing, swap to `frgfm/imagenette` (a 10-class subset, ~150 MB)

Usage:
    pip install datasets pillow
    huggingface-cli login    # if using imagenet-1k
    python build_shards_hf.py
"""
import sys
import zipfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image

# ── config ──────────────────────────────────────────────────────────────────
HF_DATASET = "ILSVRC/imagenet-1k"     # 1.28M train, gated
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
NUM_SHARDS = 4
OUT_DIR = Path("/home/pyaremenko/examples/job_imagenet_scratch/shards")
WORK = Path("/tmp/imagenet_shards_build")
MAX_EXAMPLES = None          # None = full ~150 GB; e.g. 40000 for fast smoke test
# ────────────────────────────────────────────────────────────────────────────


def _save_split(split_ds, dst: Path, label_field: str, label_names, file_filter=None) -> int:
    written = 0
    for cls in label_names:
        (dst / cls).mkdir(parents=True, exist_ok=True)
    for idx, ex in enumerate(split_ds):
        if file_filter is not None and idx not in file_filter:
            continue
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        cls_idx = ex[label_field]
        cls_name = label_names[cls_idx]
        img.save(dst / cls_name / f"{idx:08d}.jpg", "JPEG", quality=85)
        written += 1
    return written


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET}")
    ds = load_dataset(HF_DATASET)

    train_ds = ds[TRAIN_SPLIT]
    val_ds = ds.get(VAL_SPLIT) or ds.get("test")
    if MAX_EXAMPLES:
        train_ds = train_ds.select(range(min(MAX_EXAMPLES, len(train_ds))))

    label_field = "label"
    label_feat = train_ds.features[label_field]
    label_names = list(label_feat.names) if hasattr(label_feat, "names") else None
    if label_names is None:
        label_names = sorted({str(ex[label_field]) for ex in train_ds})
    print(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0} | Classes: {len(label_names)}")

    indices = list(range(len(train_ds)))
    partitions = [set(indices[i::NUM_SHARDS]) for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        n_train = _save_split(train_ds, shard_root / "images" / "train", label_field, label_names, file_filter=part)
        n_val = _save_split(val_ds, shard_root / "images" / "val", label_field, label_names) if val_ds else 0

        with (shard_root / "data.yaml").open("w") as f:
            f.write(f"nc: {len(label_names)}\n")
            f.write(f"names: {label_names}\n")
        print(f"  shard_{i}: train={n_train} val={n_val}")

        zip_path = OUT_DIR / f"shard_{i}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in shard_root.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(shard_root))
        print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    sys.exit(main())
