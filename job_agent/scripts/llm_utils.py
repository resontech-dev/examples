"""Generic Unsloth + LoRA + NVFlare adapter — fine-tunes a HuggingFace causal
LM on agent-style (function-calling) data using Unsloth's optimized kernels.

Why LoRA: full LLM fine-tuning would mean shipping ~GB of weights between
server and clients each round. LoRA trains tiny adapter matrices (~5-10 MB
at rank 16) that piggyback on a frozen base model. The base lives only on
each worker; only the LoRA delta flows through FL.

Public API:
    load_base_with_lora(model_id, ...)       -> (peft_model, tokenizer)
    train(model, tokenizer, dataset, ...)    -> LoRA state_dict (cpu)
    make_fl_adapter(model_id, ...)           -> fl_train_model
"""
# IMPORTANT: Unsloth must be imported BEFORE transformers/peft so its
# monkey-patches take effect. Do not reorder these imports.
import unsloth  # noqa: F401 — side-effect import, must come first
from unsloth import FastLanguageModel

import os

import torch

# Compat shim for old checkpoints
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{"weights_only": False, **kw})

from datasets import Dataset
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_base_with_lora(
    model_id: str,
    lora_rank: int = 16,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    target_modules=None,
):
    """Load base via Unsloth and wrap with LoRA in one step."""
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,                   # auto-detect (bf16 on supported HW)
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        base,
        r=lora_rank,
        target_modules=target_modules
        or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        # use_gradient_checkpointing=True uses HF's standard recompute (no
        # CPU offload). "unsloth" is the offloading variant — it cuts VRAM
        # but forces CPU↔GPU traffic every backward, which is what made
        # steps take 17 s on a 24 GB GPU. We have plenty of VRAM, skip it.
        use_gradient_checkpointing=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    """Pre-tokenize so the Trainer sees ready-to-batch token ids. Done with
    num_proc=1 to side-step the multiprocessing-pickle issues that surface
    when SFTTrainer/formatting_func closures capture torch/unsloth state."""
    def fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(
        fn,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Tokenizing",
    )


def train(model, tokenizer, dataset: Dataset, out_dir: str, epochs: int, batch_size: int, lr: float, max_length: int = 1024, grad_accum: int = 8) -> dict:
    """Train a LoRA-wrapped causal LM on a single shard, return LoRA state_dict.

    Uses plain HF Trainer + DataCollatorForLanguageModeling — bypasses TRL's
    SFTTrainer entirely to avoid its pre-processing pickle issues.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenized = _tokenize_dataset(dataset, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "run"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # Plain (not paged) 8-bit AdamW is faster — paged variant pages
        # optimizer state to CPU RAM and only matters when you're truly
        # tight on VRAM. We're not.
        optim="adamw_8bit",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    return {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(model).items()}


def make_fl_adapter(
    model_id: str,
    lora_rank: int = 16,
    max_seq_length: int = 1024,
    max_batch_size: int = 2,
    max_lr: float = 2e-4,
    grad_accum: int = 8,
    target_modules=None,
):
    """Build fl_train_model for an LLM LoRA FL job using Unsloth.

    LLM-specific knobs are set HERE (not in config_fed_client.json) because
    the platform's defaults are sized for vision jobs:
      - batch_size=32  → OOMs a 7B LLM, clamp to max_batch_size
      - learning_rate=1e-3 → explodes LoRA gradients, clamp to max_lr

    Usage in model_def.py:
        from llm_utils import make_fl_adapter
        fl_train_model = make_fl_adapter("unsloth/Llama-3.2-1B-bnb-4bit")
    """

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        model, tokenizer = load_base_with_lora(
            model_id, lora_rank=lora_rank, max_seq_length=max_seq_length,
            target_modules=target_modules,
        )

        if payload.get("initial_weights"):
            tensor_weights = {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()}
            try:
                set_peft_model_state_dict(model, tensor_weights)
            except Exception as e:
                if logger:
                    logger({"msg": f"set_peft_model_state_dict failed (round 0?): {e}"})

        dataset = Dataset.from_json(os.path.join(data_root, "data.jsonl"))

        # Clamp generic FL config defaults that would break LLM training.
        bs = min(int(env["batch_size"]), max_batch_size)
        lr = min(float(env.get("learning_rate", max_lr)), max_lr)

        weights = train(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=bs,
            lr=lr,
            max_length=max_seq_length,
            grad_accum=grad_accum,
        )

        return {"weights": weights, "samples": len(dataset)}

    return fl_train_model
