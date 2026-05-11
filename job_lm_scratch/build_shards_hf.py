"""Build LM-pretraining FL shards from a tokenizable text corpus.

Defaults to a small-but-real subset of The Pile. For full Pythia-scale
training (300B tokens), the standard route is to use EleutherAI's
already-tokenized data hosted on HF — see notes in `model_def.py`.

For development scale, this script downloads a public small corpus, splits
into 4 shards as JSONL with `{"text": "..."}` rows, and zips.

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
HF_DATASET = "monology/pile-uncopyrighted"     # Pile minus copyrighted books, public
TRAIN_SPLIT = "train"
NUM_SHARDS = 4
OUT_DIR = Path("/home/pyaremenko/examples/job_lm_scratch/shards")
WORK = Path("/tmp/lm_shards_build")
MAX_EXAMPLES = 4000           # small smoke-test default; bump or set None for real runs
TEXT_FIELD = "text"
# ────────────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split=TRAIN_SPLIT, streaming=False)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))
    print(f"Total: {len(ds)} text rows")

    indices = list(range(len(ds)))
    partitions = [indices[i::NUM_SHARDS] for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        shard_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = shard_root / "data.jsonl"

        with jsonl_path.open("w") as f:
            for idx in part:
                txt = ds[idx][TEXT_FIELD]
                if not txt:
                    continue
                f.write(json.dumps({"text": txt}) + "\n")
        print(f"  shard_{i}: {len(part)} rows")

        zip_path = OUT_DIR / f"shard_{i}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(jsonl_path, "data.jsonl")
        print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
