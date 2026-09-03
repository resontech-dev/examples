"""
User-owned module shipped verbatim to every worker by the ResonTech SDK.

Three named entry points the SDK looks up here:

  * ``Model``        — your ``torch.nn.Module`` subclass.
  * ``fl_train``     — one local round of training. REQUIRED.
  * ``fl_validate``  — one validation pass. OPTIONAL (skipped if missing).

Replace the model class, dataset loader, and training loop with your own;
the SDK's contract is just the function signatures and return shape below.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ── Model ────────────────────────────────────────────────────────────────────


class Model(nn.Module):
    """
    Tiny CNN for 28×28 grayscale images (MNIST-style).

    Swap the architecture freely — the only requirement is that the class
    constructor accepts whatever you put in ``ModelConfig.model_args``
    (declared in submit.py).
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ── Dataset helper ───────────────────────────────────────────────────────────


def _load_split(data_root: str, split: str) -> TensorDataset:
    """
    Shards are produced by ``build_shards.py`` and contain ``train.pt`` and
    ``val.pt`` files at the shard root. The worker mounts the unpacked shard
    at ``env["DATA_ROOT"]`` (default: ``/app/payload``).

    Swap this for whatever your shard layout is — e.g. an ``ImageFolder`` over
    ``data_root/images/`` and ``data_root/labels/``.
    """
    path = os.path.join(data_root, f"{split}.pt")
    payload = torch.load(path, map_location="cpu")
    return TensorDataset(payload["x"], payload["y"])


# ── Training (REQUIRED) ──────────────────────────────────────────────────────


def fl_train(
    model: Model,
    env: dict[str, Any],
    out_dir: str,
    logger: Any = None,
) -> dict[str, Any]:
    """
    One local training round.

    Parameters
    ----------
    model : Model
        Fresh instance built by the executor from your ``ModelConfig`` —
        already loaded with the latest global weights for this round.
    env : dict
        Merged ``TrainingConfig`` (fixed fields + ``extra``) plus
        ``DATA_ROOT`` pointing at the unpacked shard. Known keys here:
            ``EPOCHS``      → local epochs per round
            ``BATCH_SIZE``  → dataloader batch size
            ``LR``          → learning rate
            ``DATA_ROOT``   → path to this worker's shard contents
            ``weight_decay``, ``num_workers`` → any extra you set in submit.py
    out_dir : str
        Worker-local scratch dir — write per-round artefacts here if useful
        (the SDK does not require anything to land here).
    logger : optional
        NVFlare logger when available, otherwise ``None``.

    Returns
    -------
    dict with at minimum:
        ``"weights"``  → state dict to aggregate (PyTorch ``Tensor``s)
        ``"samples"``  → int, number of training samples in this round
    Extra keys are passed through to the aggregator's metadata.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    train_set = _load_split(env["DATA_ROOT"], "train")
    train_loader = DataLoader(
        train_set,
        batch_size=int(env["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(env.get("num_workers", 0)),
        pin_memory=device == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(env["LR"]),
        weight_decay=float(env.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(env["EPOCHS"])
    n_samples = 0
    for epoch in range(epochs):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            n_samples += x.size(0)
        avg = running / max(len(train_set), 1)
        msg = f"[fl_train] epoch {epoch + 1}/{epochs} avg_loss={avg:.4f}"
        (logger.info if logger else print)(msg)

    return {
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "samples": n_samples // max(epochs, 1),
    }


# ── Validation (OPTIONAL) ────────────────────────────────────────────────────


def fl_validate(
    model: Model,
    env: dict[str, Any],
    out_dir: str,
    logger: Any = None,
) -> dict[str, Any]:
    """
    One validation pass. Same signature as ``fl_train``.

    Returns
    -------
    dict with at minimum:
        ``"metrics"``  → free-form dict of scalars (acc, loss, ...)
        ``"samples"``  → int, number of validation samples
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    val_set = _load_split(env["DATA_ROOT"], "val")
    val_loader = DataLoader(val_set, batch_size=int(env["BATCH_SIZE"]))

    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += x.size(0)

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    msg = f"[fl_validate] acc={acc:.4f} loss={avg_loss:.4f} n={total}"
    (logger.info if logger else print)(msg)

    return {
        "metrics": {"accuracy": acc, "loss": avg_loss},
        "samples": total,
    }
