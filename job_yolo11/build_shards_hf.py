"""Build COCO 2017 detection FL shards from HuggingFace.

COCO is THE canonical object detection benchmark — 80 classes, 118k train /
5k val images. Real-scale training task (not a 1k-image niche).

Layout matches yolo11_utils.py expectations:
    shard_X.zip
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   (YOLO format, normalized cx cy w h)
        labels/val/*.txt
        data.yaml            (nc=80, names=[...])

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
HF_DATASET = "rafaelpadilla/coco2017"   # ~118k train / 5k val, 80 classes
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
NUM_SHARDS = 4
OUT_DIR = Path("/home/pyaremenko/examples/job_yolo11/shards")
WORK = Path("/tmp/coco_shards_build")
MAX_EXAMPLES = None                     # None = full COCO; set to 4000 for smoke test
# ────────────────────────────────────────────────────────────────────────────


def _bbox_to_yolo(bbox, w, h, fmt="xywh"):
    if fmt == "xyxy":
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        x, y = x1, y1
    else:
        x, y, bw, bh = bbox
    return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h


def _materialize(split_ds, dst_images: Path, dst_labels: Path, file_filter=None) -> tuple[int, list[str]]:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    feat = split_ds.features["objects"].feature["category"]
    class_names = list(feat.names) if hasattr(feat, "names") else None

    written = 0
    for idx, ex in enumerate(split_ds):
        if file_filter is not None and idx not in file_filter:
            continue
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        stem = f"{ex.get('image_id', idx):012d}"
        img.save(dst_images / f"{stem}.jpg", "JPEG", quality=90)

        boxes = ex["objects"]["bbox"]
        cats = ex["objects"]["category"]
        lines = []
        for box, cls in zip(boxes, cats):
            cx, cy, nw, nh = _bbox_to_yolo(box, w, h)
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with (dst_labels / f"{stem}.txt").open("w") as f:
            f.write("\n".join(lines))
        written += 1
    return written, class_names or []


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET}")
    ds = load_dataset(HF_DATASET)
    train_ds = ds[TRAIN_SPLIT]
    val_ds = ds.get(VAL_SPLIT) or ds.get("validation")
    if MAX_EXAMPLES:
        train_ds = train_ds.select(range(min(MAX_EXAMPLES, len(train_ds))))
    print(f"Train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")

    indices = list(range(len(train_ds)))
    partitions = [set(indices[i::NUM_SHARDS]) for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        n_train, class_names = _materialize(
            train_ds, shard_root / "images" / "train", shard_root / "labels" / "train",
            file_filter=part,
        )
        n_val = 0
        if val_ds:
            n_val, _ = _materialize(val_ds, shard_root / "images" / "val", shard_root / "labels" / "val")

        with (shard_root / "data.yaml").open("w") as f:
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")
        print(f"  shard_{i}: train={n_train} val={n_val} nc={len(class_names)}")

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
