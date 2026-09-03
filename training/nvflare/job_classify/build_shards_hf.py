"""Build classification FL shards from a HuggingFace image-classification dataset.

Tested with: `cifar100`, `food101`, `oxford_flowers`, `dpdl-benchmark/oxford_flowers102`.

Layout: shard contents are `images/{train,val}/<class_name>/*.jpg` so they
can be loaded by `torchvision.datasets.ImageFolder` directly. data.yaml
gives the runtime adapter the class count.

Usage:
    pip install datasets pillow
    python build_shards_hf.py
"""
import sys
import zipfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image

# ── config ──────────────────────────────────────────────────────────────────
HF_DATASET = "ethz/food101"        # 101 classes, 75,750 train / 25,250 test, 224x224
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
NUM_SHARDS = 4
OUT_DIR = Path(__file__).resolve().parent / "shards"
WORK = Path("/tmp/classify_shards_build")
MAX_EXAMPLES = None                # None = full
# ────────────────────────────────────────────────────────────────────────────


def _label_names(ds):
    feat = ds.features
    label_field = "fine_label" if "fine_label" in feat else "label"
    return feat[label_field].names, label_field


def _save_split(split_ds, dst: Path, label_names, label_field: str, file_filter=None) -> int:
    written = 0
    for cls in label_names:
        (dst / cls).mkdir(parents=True, exist_ok=True)
    for idx, ex in enumerate(split_ds):
        if file_filter is not None and idx not in file_filter:
            continue
        img = ex["img"] if "img" in ex else ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        cls_name = label_names[ex[label_field]]
        img.save(dst / cls_name / f"{idx:08d}.jpg", "JPEG", quality=85)
        written += 1
    return written


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF dataset: {HF_DATASET}")
    ds = load_dataset(HF_DATASET)
    print(f"Splits: {list(ds.keys())}")

    train_ds = ds[TRAIN_SPLIT]
    val_ds = ds.get(VAL_SPLIT) or ds.get("validation") or ds.get("test")
    if MAX_EXAMPLES:
        train_ds = train_ds.select(range(min(MAX_EXAMPLES, len(train_ds))))

    label_names, label_field = _label_names(train_ds)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0} | Classes: {len(label_names)}")

    indices = list(range(len(train_ds)))
    partitions = [set(indices[i::NUM_SHARDS]) for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        n_train = _save_split(train_ds, shard_root / "images" / "train", label_names, label_field, file_filter=part)
        n_val = _save_split(val_ds, shard_root / "images" / "val", label_names, label_field) if val_ds else 0

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
        print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
