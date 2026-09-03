"""Generic MONAI + NVFlare adapter for medical imaging classification FL jobs.

This adapter targets the MedNIST classification task (6 medical image
modalities) as a friendly intro to MONAI + FL. For real medical imaging
(CT, MRI, pathology), swap the dataset and model class.

Public API:
    monai_model(num_classes, pretrained_id) -> nn.Module (MONAI DenseNet)
    train(model, data_root, ...)            -> state_dict
    make_fl_adapter(num_classes=6, ...)     -> (yolo_model, fl_train_model)

Shard layout (same as job_classify):
    shard_X.zip
        images/train/<class_name>/*.jpg
        images/val/<class_name>/*.jpg
        data.yaml
"""
import os

import torch
import torch.nn as nn
import yaml as _yaml

from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from monai.data import DataLoader, ImageDataset


def monai_model(num_classes: int = 6, pretrained_id: str = None, **kwargs):
    """Return MONAI DenseNet121 — the canonical baseline for medical image
    classification. `pretrained_id` is ignored (DenseNet121 in MONAI uses
    its own ImageNet-init path)."""
    return DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes)


def _list_images(images_root: str):
    """Walk an ImageFolder-style tree and return (paths, labels, class_names)."""
    paths, labels = [], []
    class_names = sorted(os.listdir(images_root))
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    for cls in class_names:
        cls_dir = os.path.join(images_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(cls_dir, f))
                labels.append(cls_to_idx[cls])
    return paths, labels, class_names


def _make_loader(images_root: str, batch_size: int, train: bool):
    paths, labels, _ = _list_images(images_root)
    if train:
        tf = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize((64, 64)),
            RandRotate(range_x=15.0, prob=0.5),
            RandFlip(spatial_axis=0, prob=0.5),
            ToTensor(),
        ])
    else:
        tf = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize((64, 64)),
            ToTensor(),
        ])
    ds = ImageDataset(image_files=paths, labels=labels, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True), len(ds)


def train(model, data_root: str, out_dir: str, epochs: int, batch_size: int, lr: float) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    loader, _ = _make_loader(os.path.join(data_root, "images", "train"), batch_size, train=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True).long()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step % 50 == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f}", flush=True)

    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def _read_class_count(data_root: str) -> int:
    yaml_path = os.path.join(data_root, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            return int(_yaml.safe_load(f)["nc"])
    return len(os.listdir(os.path.join(data_root, "images", "train")))


def make_fl_adapter(num_classes: int = None, max_batch_size: int = 32, max_lr: float = 1e-3):
    """Build (yolo_model, fl_train_model) for a MONAI medical-imaging FL job."""

    def yolo_model(**kwargs):
        return monai_model(num_classes or 6)

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        nc = num_classes or _read_class_count(data_root)
        model = monai_model(nc)

        if payload.get("initial_weights"):
            current = model.state_dict()
            incoming = {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()}
            compat = {k: v for k, v in incoming.items() if k in current and current[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)

        bs = min(int(env["batch_size"]), max_batch_size)
        lr = min(float(env.get("learning_rate", max_lr)), max_lr)

        weights = train(
            model=model,
            data_root=data_root,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=bs,
            lr=lr,
        )

        train_dir = os.path.join(data_root, "images", "train")
        samples = sum(
            1 for root, _, files in os.walk(train_dir)
            for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        return {"weights": weights, "samples": samples}

    return yolo_model, fl_train_model
