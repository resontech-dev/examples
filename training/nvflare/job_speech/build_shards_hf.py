"""Build Whisper ASR FL shards from LibriSpeech (the canonical ASR benchmark).

LibriSpeech-100 = 100h `train-clean-100` subset, ~28,539 clips. Real-scale
ASR fine-tuning (not a 500-clip toy like MINDS-14).

Each shard contains audio/*.wav + data.jsonl mapping audio paths to transcripts.

Usage:
    pip install datasets librosa soundfile
    python build_shards_hf.py
"""
import json
import sys
import zipfile
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

# ── config ──────────────────────────────────────────────────────────────────
HF_DATASET = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"               # subset: clean / other
TRAIN_SPLIT = "train.100"               # 100h subset, 28,539 clips
NUM_SHARDS = 4
OUT_DIR = Path(__file__).resolve().parent / "shards"
WORK = Path("/tmp/speech_shards_build")
TEXT_FIELD = "text"                    # column name with the transcript
MAX_EXAMPLES = None                    # None = full 100h; e.g. 4000 for fast smoke test
# ────────────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {HF_DATASET} ({DATASET_CONFIG}, {TRAIN_SPLIT})")
    ds = load_dataset(HF_DATASET, DATASET_CONFIG, split=TRAIN_SPLIT)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))
    print(f"Total: {len(ds)} clips (~{len(ds) * 8 / 3600:.0f}h estimated)")

    indices = list(range(len(ds)))
    partitions = [indices[i::NUM_SHARDS] for i in range(NUM_SHARDS)]

    for i, part in enumerate(partitions):
        shard_root = WORK / f"shard_{i}"
        audio_dir = shard_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = shard_root / "data.jsonl"

        with jsonl_path.open("w") as out_f:
            for j, idx in enumerate(part):
                ex = ds[idx]
                audio = ex["audio"]
                arr = audio["array"]
                sr = audio["sampling_rate"]
                rel = f"audio/{j:08d}.wav"
                sf.write(audio_dir / f"{j:08d}.wav", arr, sr, subtype="PCM_16")
                out_f.write(json.dumps({"audio": rel, "text": ex[TEXT_FIELD]}) + "\n")
        print(f"  shard_{i}: {len(part)} clips")

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
