"""Generic timm + NVFlare adapter for image classification FL jobs.

Public API:
    timm_model(num_classes, pretrained_id) -> nn.Module
    train(model, data_root, ...)            -> state_dict (cpu floats)
    make_fl_adapter(pretrained_id, num_classes, imgsz=224)
                                            -> (yolo_model, fl_train_model)

Shard layout expected (same convention as job_csgo):
    shard_X.zip
        images/train/<class_name>/*.jpg
        images/val/<class_name>/*.jpg
        data.yaml          # {nc: N, names: [c0, c1, ...]}
"""
import os

import torch
import torch.nn as nn
import yaml as _yaml

import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def timm_model(num_classes: int, pretrained_id: str = "resnet50.a1_in1k", **kwargs):
    """Returns a timm model with `num_classes` output head, loaded from HF Hub."""
    return timm.create_model(pretrained_id, pretrained=True, num_classes=num_classes)


def _make_loaders(data_root: str, imgsz: int, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root, "images", "train"), transform=train_tf)
    val_ds_path = os.path.join(data_root, "images", "val")
    val_ds = datasets.ImageFolder(val_ds_path, transform=val_tf) if os.path.isdir(val_ds_path) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True) if val_ds else None
    return train_loader, val_loader, len(train_ds)


def train(model, data_root: str, out_dir: str, epochs: int, batch_size: int, lr: float, imgsz: int = 224) -> dict:
    """Plain PyTorch training loop. Returns state_dict (cpu)."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    train_loader, _val_loader, _n = _make_loaders(data_root, imgsz, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
    """Read nc from the shard's data.yaml."""
    yaml_path = os.path.join(data_root, "data.yaml")
    with open(yaml_path) as f:
        return int(_yaml.safe_load(f)["nc"])


def make_fl_adapter(pretrained_id: str, num_classes: int = None, imgsz: int = 224):
    """Build (yolo_model, fl_train_model) for a timm classification FL job.

    `num_classes` can be None — in which case it's read from the shard's
    data.yaml at runtime (more robust to dataset changes).
    """

    def yolo_model(**kwargs):
        # Server seeds with num_classes from arg (defaults to 1000-class
        # ImageNet head when None — gets overwritten on round 0 anyway).
        return timm_model(num_classes or 1000, pretrained_id=pretrained_id)

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        nc = num_classes or _read_class_count(data_root)
        model = timm_model(nc, pretrained_id=pretrained_id)

        if payload.get("initial_weights"):
            current = model.state_dict()
            incoming = {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()}
            # Drop shape-mismatched tensors (e.g. classification head when
            # num_classes differs from pretrained).
            compat = {k: v for k, v in incoming.items() if k in current and current[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)

        weights = train(
            model=model,
            data_root=data_root,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=int(env["batch_size"]),
            lr=float(env.get("learning_rate", 1e-3)),
            imgsz=imgsz,
        )

        # Count training images for FedAvg weighting
        train_dir = os.path.join(data_root, "images", "train")
        samples = sum(
            1 for root, _, files in os.walk(train_dir)
            for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        return {"weights": weights, "samples": samples}

    return yolo_model, fl_train_model
