"""From-scratch LM pretraining (Pythia recipe) + NVFlare adapter.

Implements EleutherAI's Pythia recipe: GPTNeoX architecture, random init,
pretrained on a tokenized text corpus (The Pile or a subset). Hyperparameters
match the published Pythia configs (https://github.com/EleutherAI/pythia/blob/main/models/).

This is a corporate-grade from-scratch pretraining task — designed for
H100-class workers; not feasible on consumer GPUs at full scale.

Public API:
    build_model(config_name, vocab_size)     -> nn.Module (random init)
    train(model, data_root, ...)             -> state_dict (cpu)
    make_fl_adapter(config_name, ...)        -> (yolo_model, fl_train_model)
"""
import os

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


# Pythia-1B config (verbatim from https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1B.yml)
PYTHIA_CONFIGS = {
    "pythia-70m": dict(
        hidden_size=512, num_attention_heads=8, num_hidden_layers=6,
        intermediate_size=2048, max_position_embeddings=2048,
        vocab_size=50304,
    ),
    "pythia-160m": dict(
        hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
        intermediate_size=3072, max_position_embeddings=2048,
        vocab_size=50304,
    ),
    "pythia-410m": dict(
        hidden_size=1024, num_attention_heads=16, num_hidden_layers=24,
        intermediate_size=4096, max_position_embeddings=2048,
        vocab_size=50304,
    ),
    "pythia-1b": dict(
        hidden_size=2048, num_attention_heads=8, num_hidden_layers=16,
        intermediate_size=8192, max_position_embeddings=2048,
        vocab_size=50304,
    ),
}


def build_model(config_name: str = "pythia-1b") -> GPTNeoXForCausalLM:
    """Build a Pythia-style GPTNeoX model from scratch (random init)."""
    if config_name not in PYTHIA_CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Choose from {list(PYTHIA_CONFIGS)}")

    arch = PYTHIA_CONFIGS[config_name]
    config = GPTNeoXConfig(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        num_attention_heads=arch["num_attention_heads"],
        intermediate_size=arch["intermediate_size"],
        max_position_embeddings=arch["max_position_embeddings"],
        rotary_pct=0.25,                # Pythia default
        rotary_emb_base=10000,
        use_parallel_residual=True,     # Pythia uses GPT-J style parallel attention
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False,      # Pythia does NOT tie embeddings
        use_cache=False,
    )
    return GPTNeoXForCausalLM(config)


def _load_tokenized_shard(data_root: str, tokenizer, max_seq_length: int = 2048):
    """Load tokenized text shard. Expects either:
       - data.jsonl with `{"text": "..."}` rows (will be tokenized on-the-fly), or
       - already-tokenized parquet/arrow at `tokens/`
    """
    jsonl_path = os.path.join(data_root, "data.jsonl")
    if os.path.exists(jsonl_path):
        ds = load_dataset("json", data_files=jsonl_path, split="train")

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None,
            )

        return ds.map(
            tokenize, batched=True, batch_size=1000, num_proc=1,
            remove_columns=ds.column_names, desc="Tokenizing",
        )

    tokens_dir = os.path.join(data_root, "tokens")
    if os.path.isdir(tokens_dir):
        return load_dataset("arrow", data_files=os.path.join(tokens_dir, "*.arrow"), split="train")

    raise FileNotFoundError(f"No data.jsonl or tokens/ in {data_root}")


def train(
    model,
    data_root: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float = 6e-4,
    weight_decay: float = 0.1,
    max_seq_length: int = 2048,
    grad_accum: int = 4,
    tokenizer_name: str = "EleutherAI/pythia-1b",
) -> dict:
    """Pythia-style pretraining loop. Returns state_dict (cpu)."""
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _load_tokenized_shard(data_root, tokenizer, max_seq_length)

    # Pythia hyperparameters (https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1B.yml)
    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "run"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def make_fl_adapter(
    config_name: str = "pythia-1b",
    max_seq_length: int = 2048,
    max_batch_size: int = 4,
    max_lr: float = 6e-4,
    grad_accum: int = 4,
):
    """Build (yolo_model, fl_train_model) for from-scratch Pythia FL job.

    Usage in model_def.py:
        from pythia_utils import make_fl_adapter
        yolo_model, fl_train_model = make_fl_adapter("pythia-1b")
    """

    def yolo_model(**kwargs):
        return build_model(config_name)

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        model = build_model(config_name)

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
            max_seq_length=max_seq_length,
            grad_accum=grad_accum,
        )

        # Sample count for FedAvg weighting
        jsonl_path = os.path.join(data_root, "data.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                samples = sum(1 for _ in f)
        else:
            samples = 1
        return {"weights": weights, "samples": samples}

    return yolo_model, fl_train_model
