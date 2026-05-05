"""Generic LoRA + transformers + NVFlare adapter — usable in any FL job that
fine-tunes a HuggingFace causal LM via LoRA.

Why LoRA: full LLM fine-tuning would mean shipping ~GB of weights between
server and clients each round. LoRA trains tiny adapter matrices (typically
~5-10 MB at rank 16) that piggyback on a frozen base model. Each worker
holds the base locally; only the LoRA delta flows through FL.

Public API:
    load_base(model_id)                  -> (model, tokenizer)
    apply_lora(model, rank=16)           -> peft model
    train(model, tokenizer, ...)         -> LoRA state_dict (cpu)
    make_fl_adapter(model_id, ...)       -> fl_train_model
"""
import json
import os

import torch

# Suppress old-checkpoint compat noise
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{"weights_only": False, **kw})

from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_base(model_id: str):
    """Download (or use cache) and load the base causal LM in fp16."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    return model, tokenizer


def apply_lora(model, rank: int = 16, target_modules=None):
    """Wrap a causal LM with a LoRA adapter."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=target_modules,  # None = peft auto-detect
    )
    return get_peft_model(model, config)


def _load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _format_alpaca(ex: dict) -> str:
    """Standard Alpaca prompt template."""
    instr = ex.get("instruction", "")
    inp = ex.get("input") or ex.get("context") or ""
    out = ex.get("output") or ex.get("response") or ""
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"


def train(model, tokenizer, dataset: Dataset, out_dir: str, epochs: int, batch_size: int, lr: float, max_length: int = 512) -> dict:
    """Run SFT on a single shard of instruction data, return LoRA state_dict."""
    os.makedirs(out_dir, exist_ok=True)

    config = SFTConfig(
        output_dir=os.path.join(out_dir, "run"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        max_length=max_length,
        gradient_accumulation_steps=1,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        formatting_func=_format_alpaca,
    )
    trainer.train()

    return {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(model).items()}


def make_fl_adapter(model_id: str, lora_rank: int = 16, max_length: int = 512, target_modules=None):
    """Build fl_train_model for an LLM LoRA FL job in one call.

    Usage in model_def.py:
        from llm_utils import make_fl_adapter
        fl_train_model = make_fl_adapter("HuggingFaceTB/SmolLM2-360M")
    """

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        # 1. Load base + apply fresh LoRA
        base, tokenizer = load_base(model_id)
        model = apply_lora(base, rank=lora_rank, target_modules=target_modules)

        # 2. Apply incoming round's aggregated LoRA weights (if any)
        if payload.get("initial_weights"):
            tensor_weights = {
                k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()
            }
            try:
                set_peft_model_state_dict(model, tensor_weights)
            except Exception as e:
                if logger:
                    logger({"msg": f"set_peft_model_state_dict failed (round 0?): {e}"})

        # 3. Load this shard's instruction data
        examples = _load_jsonl(os.path.join(data_root, "data.jsonl"))
        dataset = Dataset.from_list(examples)

        # 4. Train via SFT
        weights = train(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=int(env["batch_size"]),
            lr=float(env.get("learning_rate", 1e-4)),
            max_length=max_length,
        )

        return {"weights": weights, "samples": len(dataset)}

    return fl_train_model
