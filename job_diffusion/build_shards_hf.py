"""Build SD-LoRA FL shards from a HuggingFace image-caption dataset.

Tested with:
- `lambdalabs/pokemon-blip-captions` (~830 captioned Pokémon, ~70 MB)
- `recmek/anime-captions`
- `m1guelpf/nouns` (artistic style)

Each shard: images/*.jpg + data.jsonl with {image, caption}.

Usage:
    pip install datasets pillow
    python build_shards_hf.py
"""
import json
import sys
import zipfile
from pathlib import Path

from datasets import load_dataset

# ── config ──────────────────────────────────────────────────────────────────
HF_DATASET = "lambdalabs/pokemon-blip-captions"
TRAIN_SPLIT = "train"
NUM_SHARDS = 4
OUT_DIR = Path("/home/pyaremenko/examples/job_diffusion/shards")
WORK = Path("/tmp/diffusion_shards_build")
MAX_EXAMPLES = None
# ────────────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split=TRAIN_SPLIT)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))
    print(f"Total: {len(ds)} captioned images")

    # Auto-detect schema
    cap_field = "text" if "text" in ds.features else "caption"
    img_field = "image"

    indices = list(range(len(ds)))
    partitions = [indices[i::NUM_SHARDS] for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        img_dir = shard_root / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = shard_root / "data.jsonl"

        with jsonl_path.open("w") as f:
            for j, idx in enumerate(part):
                ex = ds[idx]
                img = ex[img_field]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                rel = f"images/{j:08d}.jpg"
                img.save(img_dir / f"{j:08d}.jpg", "JPEG", quality=90)
                f.write(json.dumps({"image": rel, "caption": ex[cap_field]}) + "\n")
        print(f"  shard_{i}: {len(part)} pairs")

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
