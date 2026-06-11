"""
Build one zip per worker, each containing a ``train.pt`` + ``val.pt``
tensor pair, ready to be uploaded by the SDK.

Defaults to MNIST (auto-downloaded by torchvision the first time, ~12 MB).
Swap the ``_load_dataset`` body for your own data — anything that produces
``(x_train, y_train, x_val, y_val)`` torch tensors works.

Usage
-----
    python build_shards.py --num-shards 3 --out-dir ./shards
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms


def _load_dataset(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_train, y_train, x_val, y_val) as torch tensors."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    train = datasets.MNIST(str(cache_dir), train=True, download=True, transform=tfm)
    val = datasets.MNIST(str(cache_dir), train=False, download=True, transform=tfm)

    x_train = torch.stack([train[i][0] for i in range(len(train))])
    y_train = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
    x_val = torch.stack([val[i][0] for i in range(len(val))])
    y_val = torch.tensor([val[i][1] for i in range(len(val))], dtype=torch.long)
    return x_train, y_train, x_val, y_val


def _stratified_split(y: torch.Tensor, num_shards: int) -> list[torch.Tensor]:
    """
    Stratified per-class split — each shard sees a roughly equal slice of
    every class. Returns one tensor of train-sample indices per shard.
    """
    indices_per_shard: list[list[int]] = [[] for _ in range(num_shards)]
    for cls in y.unique().tolist():
        cls_idx = (y == cls).nonzero(as_tuple=True)[0].tolist()
        for i, idx in enumerate(cls_idx):
            indices_per_shard[i % num_shards].append(idx)
    return [torch.tensor(sorted(s), dtype=torch.long) for s in indices_per_shard]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shards", type=int, required=True, help="One shard per worker.")
    parser.add_argument("--out-dir", type=Path, default=Path("./shards"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./.dataset_cache"))
    args = parser.parse_args()

    if args.num_shards < 1:
        sys.exit("--num-shards must be ≥ 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[shards] loading dataset (cache: {args.cache_dir})…")
    x_train, y_train, x_val, y_val = _load_dataset(args.cache_dir)
    print(f"[shards] train={len(y_train)}  val={len(y_val)}  classes={int(y_train.max()) + 1}")

    # Each shard gets a disjoint slice of the train set and the FULL val set
    # (so per-worker validation metrics are comparable round-to-round).
    train_idx_per_shard = _stratified_split(y_train, args.num_shards)

    for i, idx in enumerate(train_idx_per_shard):
        staging = args.out_dir / f"_shard_{i}_staging"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            torch.save({"x": x_train[idx], "y": y_train[idx]}, staging / "train.pt")
            torch.save({"x": x_val, "y": y_val}, staging / "val.pt")
            zip_path = shutil.make_archive(
                str(args.out_dir / f"shard_{i}"), "zip", root_dir=staging,
            )
            size_mb = Path(zip_path).stat().st_size / 1024 / 1024
            print(f"[shards] {Path(zip_path).name}  train={len(idx)}  val={len(y_val)}  {size_mb:.1f} MB")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    print(f"[shards] done → {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
