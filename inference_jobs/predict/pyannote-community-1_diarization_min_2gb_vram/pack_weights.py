"""
Download the gated pyannote Community-1 pipeline with YOUR Hugging Face token
and pack it for submit.py. The token stays on this machine; the job only gets
the packed files.

Once:
  1. Accept the conditions on https://huggingface.co/pyannote/speaker-diarization-community-1
  2. Create a read token at https://huggingface.co/settings/tokens and put it in .env as HF_TOKEN

Run
---
    pip install huggingface_hub python-dotenv
    python pack_weights.py      # -> model/speaker-diarization-community-1.tar.gz (~35 MB)
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
from pathlib import Path

from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

REPO = "pyannote/speaker-diarization-community-1"
OUT = HERE / "model" / "speaker-diarization-community-1.tar.gz"


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("[pack] Missing HF_TOKEN in .env (a read token from https://huggingface.co/settings/tokens)")
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        sys.exit("[pack] pip install huggingface_hub")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = Path(snapshot_download(REPO, token=token, local_dir=Path(tmp) / "community-1"))
        except GatedRepoError:
            sys.exit(f"[pack] Access denied: accept the conditions on https://huggingface.co/{REPO} with this token's account")
        OUT.parent.mkdir(exist_ok=True)
        with tarfile.open(OUT, "w:gz") as tar:
            for p in sorted(local.rglob("*")):
                rel = p.relative_to(local)
                if p.is_file() and ".cache" not in rel.parts:
                    tar.add(p, arcname=str(Path("community-1") / rel))
    print(f"[pack] {OUT.relative_to(HERE)} ({OUT.stat().st_size / 2**20:.1f} MB): now python submit.py")


if __name__ == "__main__":
    main()
