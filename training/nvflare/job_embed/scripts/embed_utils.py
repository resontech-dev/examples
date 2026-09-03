"""Generic sentence-transformers + NVFlare adapter for embedding FL jobs.

Public API:
    load_model(model_id)                     -> SentenceTransformer
    train(model, dataset, ...)               -> state_dict (cpu floats)
    make_fl_adapter(model_id, ...)           -> (yolo_model, fl_train_model)

Shard layout expected:
    shard_X.zip
        data.jsonl         # one line per example:
                          # {"anchor": str, "positive": str}     # positive pairs
                          # OR
                          # {"anchor": str, "positive": str, "negative": str}  # triplets
"""
import json
import os

import torch

# Compat shim
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{"weights_only": False, **kw})

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader


def load_model(model_id: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    return SentenceTransformer(model_id)


def _load_examples(jsonl_path: str):
    has_negatives = False
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if "negative" in row:
                examples.append(InputExample(texts=[row["anchor"], row["positive"], row["negative"]]))
                has_negatives = True
            else:
                examples.append(InputExample(texts=[row["anchor"], row["positive"]]))
    return examples, has_negatives


def train(model: SentenceTransformer, data_jsonl: str, out_dir: str, epochs: int, batch_size: int, lr: float) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    examples, has_neg = _load_examples(data_jsonl)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    if has_neg:
        loss = losses.TripletLoss(model)
    else:
        loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=int(0.05 * len(loader) * epochs),
        optimizer_params={"lr": lr},
        weight_decay=0.01,
        max_grad_norm=1.0,
        show_progress_bar=True,
        output_path=os.path.join(out_dir, "run"),
        save_best_model=False,
    )

    # SentenceTransformer is a torch.nn.Sequential of modules — flatten state_dict
    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def make_fl_adapter(model_id: str = "BAAI/bge-small-en-v1.5", max_batch_size: int = 32, max_lr: float = 2e-5):
    """Build (yolo_model, fl_train_model) for a sentence-embedding FL job."""

    def yolo_model(**kwargs):
        # Persistor needs an nn.Module — SentenceTransformer subclasses it.
        return load_model(model_id)

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        model = load_model(model_id)

        if payload.get("initial_weights"):
            current = model.state_dict()
            incoming = {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()}
            compat = {k: v for k, v in incoming.items() if k in current and current[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)

        bs = min(int(env["batch_size"]), max_batch_size)
        lr = min(float(env.get("learning_rate", max_lr)), max_lr)

        weights = train(
            model=model,
            data_jsonl=os.path.join(data_root, "data.jsonl"),
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=bs,
            lr=lr,
        )

        # Count examples for FedAvg weighting
        with open(os.path.join(data_root, "data.jsonl")) as f:
            samples = sum(1 for _ in f)
        return {"weights": weights, "samples": samples}

    return yolo_model, fl_train_model
