"""Build LLM FL shards from any HuggingFace instruction dataset that has
`instruction` + (`input` or `context`) + (`output` or `response`) fields.

Tested with: `tatsu-lab/alpaca`, `databricks/databricks-dolly-15k`, `yahma/alpaca-cleaned`.

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
HF_DATASET = "tatsu-lab/alpaca"               # 52k Alpaca instructions
TRAIN_SPLIT = "train"
NUM_SHARDS = 4
OUT_DIR = Path(__file__).resolve().parent / "shards"
WORK = Path("/tmp/llm_shards_build")
# Optional: subsample to keep training time reasonable. None = use all.
MAX_EXAMPLES = None
# ────────────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF dataset: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split=TRAIN_SPLIT)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))
    print(f"Total examples: {len(ds)}")

    indices = list(range(len(ds)))
    partitions = [indices[i::NUM_SHARDS] for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        shard_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = shard_root / "data.jsonl"

        with jsonl_path.open("w") as f:
            for idx in part:
                ex = ds[idx]
                # Keep only the fields llm_utils._format_alpaca knows about
                row = {
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input") or ex.get("context") or "",
                    "output": ex.get("output") or ex.get("response") or "",
                }
                f.write(json.dumps(row) + "\n")
        print(f"  shard_{i}: {len(part)} examples")

        zip_path = OUT_DIR / f"shard_{i}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(jsonl_path, "data.jsonl")
        print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
