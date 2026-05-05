"""Generic yolov5 + NVFlare adapter — usable in any FL job that wraps a
yolov5 HuggingFace model.

Public API:
    load_model(pretrained_id)                 -> nn.Module
    train(model, data_root, out_dir, ...)     -> state_dict (cpu floats)
    make_fl_adapter(pretrained_id, imgsz=640) -> (yolo_model, fl_train_model)

`make_fl_adapter` is the one-liner: it returns the two callables NVFlare
expects (server-side model factory and per-round trainer), already wired
together. Drop them at module scope of model_def.py and you're done.
"""
import os

import torch
import yaml as _yaml

# yolov5 7.0.x compat shims for PyTorch 2.7+
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{"weights_only": False, **kw})
torch.use_deterministic_algorithms = lambda *a, **kw: None

import yolov5
from yolov5 import train as yolov5_train


def load_model(pretrained_id: str):
    """Return yolov5's inner DetectionModel (an nn.Module).

    `yolov5.load(...)` returns AutoShape(DetectMultiBackend(DetectionModel)).
    yolov5_train.run() needs the bare DetectionModel (it accesses `.yaml`
    on it to rebuild the architecture). This walks the chain.
    """
    m = yolov5.load(pretrained_id)
    while not hasattr(m, "yaml") and hasattr(m, "model"):
        m = m.model
    return m


def _runtime_data_yaml(shard_yaml: str, data_root: str, out_dir: str) -> str:
    """Read the shard's data.yaml, set `path:` to the absolute data_root, and
    write a runtime copy. yolov5 resolves `train:` / `val:` relative to
    `path:` — without it set, it resolves relative to its install dir."""
    with open(shard_yaml) as f:
        data = _yaml.safe_load(f)
    data["path"] = data_root
    runtime = os.path.join(out_dir, "data.yaml")
    with open(runtime, "w") as f:
        _yaml.safe_dump(data, f)
    return runtime


def train(model, data_root: str, out_dir: str, epochs: int, batch_size: int, imgsz: int = 640) -> dict:
    """Drive yolov5_train.run on an in-memory model and return trained weights.

    yolov5_train.run is CLI-style: it only accepts a path to a .pt for the
    starting weights. So we save → train → read best.pt back.
    """
    os.makedirs(out_dir, exist_ok=True)
    data_yaml = _runtime_data_yaml(os.path.join(data_root, "data.yaml"), data_root, out_dir)

    init_ckpt = os.path.join(out_dir, "init.pt")
    torch.save({"model": model, "epoch": -1}, init_ckpt)

    yolov5_train.run(
        data=data_yaml,
        weights=init_ckpt,
        imgsz=imgsz,
        batch_size=batch_size,
        epochs=epochs,
        project=out_dir,
        name="run",
        exist_ok=True,
    )

    best = os.path.join(out_dir, "run", "weights", "best.pt")
    ckpt = torch.load(best, map_location="cpu")
    inner = ckpt.get("ema") or ckpt["model"]
    sd = inner.state_dict() if hasattr(inner, "state_dict") else inner
    return {k: v.float().cpu() for k, v in sd.items()}


def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )


def make_fl_adapter(pretrained_id: str, imgsz: int = 640):
    """Build (yolo_model, fl_train_model) for a yolov5 FL job in one call.

    Usage in model_def.py:
        from yolo_utils import make_fl_adapter
        yolo_model, fl_train_model = make_fl_adapter("org/your-yolov5-model")

    Each shard must contain a `data.yaml` at its root, with at minimum
    `train:`, `val:`, `nc:`, `names:` fields. `path:` is set at runtime
    to the worker's actual data_root.
    """

    def yolo_model(**kwargs):
        return load_model(pretrained_id)

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        model = load_model(pretrained_id)
        if payload.get("initial_weights"):
            model.load_state_dict(
                {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()},
                strict=False,
            )

        weights = train(
            model=model,
            data_root=data_root,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=int(env["batch_size"]),
            imgsz=imgsz,
        )

        return {
            "weights": weights,
            "samples": _count_images(os.path.join(data_root, "images", "train")),
        }

    return yolo_model, fl_train_model
