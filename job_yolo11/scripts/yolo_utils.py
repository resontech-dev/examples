"""Generic Ultralytics + NVFlare adapter — usable for any Ultralytics YOLO
job (YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / etc.) backed by a HuggingFace
checkpoint.

Public API:
    load_model(repo_id, filename)             -> ultralytics.YOLO wrapper
    train(model, data_root, out_dir, ...)     -> state_dict (cpu floats)
    make_fl_adapter(repo_id, filename, ...)   -> (yolo_model, fl_train_model)

`make_fl_adapter` is the one-liner: returns the two callables NVFlare expects
(server-side model factory + per-round trainer), already wired together.
Drop them at module scope of model_def.py and you're done.
"""
import os

import torch
import yaml as _yaml

# Compat shim for older checkpoints under PyTorch 2.6+
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{"weights_only": False, **kw})

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def load_model(repo_id: str, filename: str = "yolo11n.pt") -> YOLO:
    """Download `filename` from the HuggingFace `repo_id` and load via Ultralytics."""
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(ckpt)


def _runtime_data_yaml(shard_yaml: str, data_root: str, out_dir: str) -> str:
    """Read the shard's data.yaml, set `path` to the absolute data_root, and
    write a runtime copy. Ultralytics requires `path:` to be an absolute path
    or relative to its DATASETS_DIR — this side-steps both."""
    with open(shard_yaml) as f:
        data = _yaml.safe_load(f)
    data["path"] = data_root
    runtime = os.path.join(out_dir, "data.yaml")
    with open(runtime, "w") as f:
        _yaml.safe_dump(data, f)
    return runtime


def train(model: YOLO, data_root: str, out_dir: str, epochs: int, batch_size: int, imgsz: int = 640) -> dict:
    """Train an Ultralytics YOLO on a shard, return trained state_dict."""
    os.makedirs(out_dir, exist_ok=True)
    data_yaml = _runtime_data_yaml(os.path.join(data_root, "data.yaml"), data_root, out_dir)

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=out_dir,
        name="run",
        exist_ok=True,
        deterministic=False,
        verbose=False,
    )
    return {k: v.float().cpu() for k, v in model.model.state_dict().items()}


def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )


def make_fl_adapter(repo_id: str, filename: str = "yolo11n.pt", imgsz: int = 640):
    """Build (yolo_model, fl_train_model) for an Ultralytics FL job in one call.

    Usage in model_def.py:
        from yolo_utils import make_fl_adapter
        yolo_model, fl_train_model = make_fl_adapter("Ultralytics/YOLO11", "yolo11n.pt")
    """

    def yolo_model(**kwargs):
        return load_model(repo_id, filename).model  # inner DetectionModel

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        wrapper = load_model(repo_id, filename)

        # Rebuild the detection head to match the dataset's nc so aggregated
        # weights (head shaped for the dataset, not the pretrained nc=80) load
        # cleanly on round 2+. Without this, load_state_dict raises on the
        # cv3 head shape mismatch.
        with open(os.path.join(data_root, "data.yaml")) as f:
            data_cfg = _yaml.safe_load(f)
        nc = int(data_cfg.get("nc", len(data_cfg.get("names", []))))
        if wrapper.model.yaml.get("nc") != nc:
            from ultralytics.nn.tasks import DetectionModel
            rebuilt = DetectionModel(cfg=wrapper.model.yaml, nc=nc, verbose=False)
            rebuilt.load(wrapper.model)
            wrapper.model = rebuilt

        if payload.get("initial_weights"):
            wrapper.model.load_state_dict(
                {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()},
                strict=False,
            )

        weights = train(
            wrapper,
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
