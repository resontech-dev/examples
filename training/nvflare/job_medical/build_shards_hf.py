"""Build HAM10000 (skin lesion) FL shards from a HuggingFace dataset.

HAM10000 — Human Against Machine with 10000 training images — is a real
dermatoscopic skin lesion benchmark with 7 classes:
    akiec, bcc, bkl, df, mel, nv, vasc

A genuine medical-imaging task (not a toy like MedNIST), used in 2018
ISIC challenge baselines.

Layout produced (compatible with monai_utils.py's ImageFolder loader):
    shard_X.zip
        images/train/<class>/*.jpg
        images/val/<class>/*.jpg
        data.yaml       # {nc: 7, names: [...]}

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
HF_DATASET = "marmal88/skin_cancer"   # 13,354 dermatoscopic images, 7 classes
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
NUM_SHARDS = 4
OUT_DIR = Path(__file__).resolve().parent / "shards"
WORK = Path("/tmp/medical_shards_build")
MAX_EXAMPLES = None
# ────────────────────────────────────────────────────────────────────────────


def _save(split_ds, dst: Path, label_field: str, label_names, file_filter=None) -> int:
    written = 0
    for cls in label_names:
        (dst / cls).mkdir(parents=True, exist_ok=True)
    for idx, ex in enumerate(split_ds):
        if file_filter is not None and idx not in file_filter:
            continue
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        cls_name = label_names[ex[label_field]]
        img.save(dst / cls_name / f"{idx:08d}.jpg", "JPEG", quality=90)
        written += 1
    return written


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET}")
    ds = load_dataset(HF_DATASET)
    print(f"Splits: {list(ds.keys())}")

    train_ds = ds[TRAIN_SPLIT]
    val_ds = ds.get(VAL_SPLIT) or ds.get("test")
    if MAX_EXAMPLES:
        train_ds = train_ds.select(range(min(MAX_EXAMPLES, len(train_ds))))

    label_field = "dx" if "dx" in train_ds.features else "label"
    feat = train_ds.features[label_field]
    label_names = list(feat.names) if hasattr(feat, "names") else sorted({ex[label_field] for ex in train_ds})
    print(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0} | Classes: {label_names}")

    indices = list(range(len(train_ds)))
    partitions = [set(indices[i::NUM_SHARDS]) for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        n_train = _save(train_ds, shard_root / "images" / "train", label_field, label_names, file_filter=part)
        n_val = _save(val_ds, shard_root / "images" / "val", label_field, label_names) if val_ds else 0

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
