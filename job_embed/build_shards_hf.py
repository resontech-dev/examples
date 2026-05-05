"""Build sentence-embedding FL shards from a HuggingFace pair/triplet dataset.

Tested with:
- `sentence-transformers/all-nli` (triplets)
- `embedding-data/QQP_triplets`
- `microsoft/ms_marco` (qa pairs)

Each shard: data.jsonl with {anchor, positive, negative?} per line.

Usage:
    pip install datasets
    python build_shards_hf.py
"""
import json
import sys
import zipfile
from pathlib import Path

from datasets import load_dataset

# ── config ──────────────────────────────────────────────────────────────────
HF_DATASET = "sentence-transformers/all-nli"
DATASET_CONFIG = "triplet"            # subset: pairs / triplet / pair-class / pair-score
TRAIN_SPLIT = "train"
NUM_SHARDS = 4
OUT_DIR = Path("/home/pyaremenko/examples/job_embed/shards")
WORK = Path("/tmp/embed_shards_build")
MAX_EXAMPLES = None
# ────────────────────────────────────────────────────────────────────────────


def _normalize(ex: dict) -> dict:
    """Extract anchor/positive(/negative) from common dataset schemas."""
    if "anchor" in ex:
        return {k: ex[k] for k in ("anchor", "positive", "negative") if k in ex}
    if "query" in ex:
        return {"anchor": ex["query"], "positive": ex.get("positive", ex.get("text", ""))}
    if "sentence1" in ex and "sentence2" in ex:
        return {"anchor": ex["sentence1"], "positive": ex["sentence2"]}
    raise KeyError(f"Unknown schema: {list(ex.keys())}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET} ({DATASET_CONFIG})")
    if DATASET_CONFIG:
        ds = load_dataset(HF_DATASET, DATASET_CONFIG, split=TRAIN_SPLIT)
    else:
        ds = load_dataset(HF_DATASET, split=TRAIN_SPLIT)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))
    print(f"Total: {len(ds)} examples")

    indices = list(range(len(ds)))
    partitions = [indices[i::NUM_SHARDS] for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        shard_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = shard_root / "data.jsonl"

        with jsonl_path.open("w") as f:
            for idx in part:
                f.write(json.dumps(_normalize(ds[idx])) + "\n")
        print(f"  shard_{i}: {len(part)} examples")

        zip_path = OUT_DIR / f"shard_{i}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(jsonl_path, "data.jsonl")
        print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
