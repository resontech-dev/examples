"""From-scratch ImageNet training (timm A2 recipe) + NVFlare adapter.

Implements `ResNet Strikes Back` (https://arxiv.org/abs/2110.00476) A2 recipe:
random-init ResNet-50, LAMB optimizer, BCE loss + label smoothing, RandAugment +
Mixup + CutMix, repeated augmentation, 300 epochs, batch 2048 (across GPUs).

This is a corporate-grade from-scratch training task — designed for
H100-class workers; not feasible on consumer GPUs at full scale.

Public API:
    build_model(model_name, num_classes)     -> nn.Module (random init)
    train(model, data_root, ...)             -> state_dict (cpu)
    make_fl_adapter(model_name, ...)         -> (yolo_model, fl_train_model)
"""
import os

import torch
import torch.nn as nn
import yaml as _yaml

import timm
from timm.data import Mixup
from timm.loss import BinaryCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_model(model_name: str = "resnet50", num_classes: int = 1000) -> nn.Module:
    """Random-init timm model (no pretrained weights)."""
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def _make_loaders(data_root: str, imgsz: int, batch_size: int):
    """A2-style augmentations for from-scratch ImageNet training."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.0),  # A2: no random erasing
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(imgsz * 256 / 224)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root, "images", "train"), transform=train_tf)
    val_path = os.path.join(data_root, "images", "val")
    val_ds = datasets.ImageFolder(val_path, transform=val_tf) if os.path.isdir(val_path) else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    ) if val_ds else None
    return train_loader, val_loader, len(train_ds)


def _read_class_count(data_root: str) -> int:
    yaml_path = os.path.join(data_root, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            return int(_yaml.safe_load(f)["nc"])
    return 1000  # ImageNet default


def train(
    model,
    data_root: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float = 5e-3,
    weight_decay: float = 0.02,
    imgsz: int = 224,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.1,
    cutmix_alpha: float = 1.0,
    drop_path_rate: float = 0.05,
) -> dict:
    """timm A2 from-scratch training loop. Returns state_dict (cpu)."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply drop_path (stochastic depth) — A2 uses 0.05 for ResNet-50
    if hasattr(model, "drop_path_rate"):
        model.drop_path_rate = drop_path_rate
    model = model.to(device)
    model.train()

    train_loader, _val_loader, n_train = _make_loaders(data_root, imgsz, batch_size)
    num_classes = _read_class_count(data_root)

    # Mixup + CutMix (A2 recipe: both, switch_prob=0.5)
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )

    # Optimizer (A2: LAMB)
    optimizer = create_optimizer_v2(
        model,
        opt="lamb",
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
    )

    # LR schedule: cosine + 5-epoch warmup (A2)
    total_steps = epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps,
        warmup_t=warmup_steps,
        warmup_lr_init=lr * 1e-2,
        warmup_prefix=True,
        cycle_limit=1,
        t_in_epochs=False,
    )

    # Loss: BCE + label smoothing (A2 — replaces CE)
    criterion = BinaryCrossEntropy(smoothing=label_smoothing, target_threshold=0.2)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y = mixup_fn(x, y)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(step)
            step += 1

            if batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch} step {batch_idx}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr_now:.2e}", flush=True)

    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def make_fl_adapter(
    model_name: str = "resnet50",
    imgsz: int = 224,
    lr: float = 5e-3,
    weight_decay: float = 0.02,
    label_smoothing: float = 0.1,
    drop_path_rate: float = 0.05,
):
    """Build (yolo_model, fl_train_model) for from-scratch ImageNet FL job.

    Usage in model_def.py:
        from imagenet_utils import make_fl_adapter
        yolo_model, fl_train_model = make_fl_adapter("resnet50")
    """

    def yolo_model(**kwargs):
        return build_model(model_name, num_classes=1000)

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        nc = _read_class_count(data_root)
        model = build_model(model_name, num_classes=nc)

        if payload.get("initial_weights"):
            current = model.state_dict()
            incoming = {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()}
            compat = {k: v for k, v in incoming.items() if k in current and current[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)

        weights = train(
            model=model,
            data_root=data_root,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=int(env["batch_size"]),
            lr=float(env.get("learning_rate", lr)),
            weight_decay=weight_decay,
            imgsz=imgsz,
            label_smoothing=label_smoothing,
            drop_path_rate=drop_path_rate,
        )

        train_dir = os.path.join(data_root, "images", "train")
        samples = sum(
            1 for root, _, files in os.walk(train_dir)
            for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        return {"weights": weights, "samples": samples}

    return yolo_model, fl_train_model
