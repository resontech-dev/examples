"""Build agent FL shards from a HuggingFace function-calling dataset.

Default: `glaiveai/glaive-function-calling-v2` (113k public examples).
Each example has `system` (function definitions) and `chat` (USER/ASSISTANT
conversation). We concatenate them into a single `text` field per example
so SFTTrainer can train directly on the formatted prompt.

To swap dataset, edit HF_DATASET below. The script will try common field
names and emit a `text` field for each example.

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
HF_DATASET = "glaiveai/glaive-function-calling-v2"
TRAIN_SPLIT = "train"
NUM_SHARDS = 4
OUT_DIR = Path(__file__).resolve().parent / "shards"
WORK = Path("/tmp/agent_shards_build")
MAX_EXAMPLES = None    # None = full; int = subsample (faster builds)
# ────────────────────────────────────────────────────────────────────────────


def _to_text(ex: dict) -> str:
    """Coerce common function-calling schemas into a single training prompt."""
    # glaiveai/glaive-function-calling-v2: system + chat
    if "system" in ex and "chat" in ex:
        return f"{ex['system'].strip()}\n\n{ex['chat'].strip()}"
    # xlam-style: tools + query + answers
    if "tools" in ex and "query" in ex:
        return (
            f"### Available tools:\n{ex['tools']}\n\n"
            f"### User:\n{ex['query']}\n\n"
            f"### Tool calls:\n{ex.get('answers', '')}"
        )
    # alpaca-style fallback
    instr = ex.get("instruction", "")
    inp = ex.get("input") or ex.get("context") or ""
    out = ex.get("output") or ex.get("response") or ""
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"


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
                f.write(json.dumps({"text": _to_text(ds[idx])}) + "\n")
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
