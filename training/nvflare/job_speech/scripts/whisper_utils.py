"""Generic Whisper + LoRA + NVFlare adapter for ASR FL jobs.

Public API:
    load_base_with_lora(model_id, ...)       -> (peft_model, processor)
    train(model, processor, dataset, ...)    -> LoRA state_dict (cpu)
    make_fl_adapter(model_id, language, ...) -> fl_train_model

Shard layout expected:
    shard_X.zip
        audio/<id>.wav     # 16 kHz mono
        data.jsonl         # one line per example: {"audio": "audio/<id>.wav", "text": "transcript"}
"""
import json
import os

import torch

# Compat shim
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
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def load_base_with_lora(model_id: str, language: str = "en", task: str = "transcribe", lora_rank: int = 16):
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.config.suppress_tokens = []

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model, processor


def _prepare_examples(examples, processor, data_root: str, max_label_len: int = 448):
    """Convert raw audio paths + text into Whisper input_features + labels."""
    import librosa
    audios = []
    for path in examples["audio"]:
        full = os.path.join(data_root, path)
        wav, sr = librosa.load(full, sr=16000)
        audios.append(wav)

    inputs = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="np")
    labels = processor.tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_label_len,
        return_tensors="np",
    )
    return {
        "input_features": inputs.input_features,
        "labels": labels.input_ids,
    }


class _DataCollatorASR:
    """Right-pad input_features and replace pad token in labels with -100."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def train(model, processor, dataset, data_root: str, out_dir: str, epochs: int, batch_size: int, lr: float) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    tokenized = dataset.map(
        lambda b: _prepare_examples(b, processor, data_root),
        batched=True,
        batch_size=8,
        num_proc=1,
        remove_columns=dataset.column_names,
        desc="Preparing audio",
    )

    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(out_dir, "run"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        fp16=True,
        predict_with_generate=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    collator = _DataCollatorASR(processor)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    return {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(model).items()}


def make_fl_adapter(model_id: str = "openai/whisper-small", language: str = "en", lora_rank: int = 16, max_batch_size: int = 4, max_lr: float = 2e-4):
    """Build fl_train_model for a Whisper ASR FL job."""

    def fl_train_model(payload, out_dir, logger=None):
        env = payload["env"]
        data_root = payload["dataset"]["data_root"]

        model, processor = load_base_with_lora(model_id, language=language, lora_rank=lora_rank)

        if payload.get("initial_weights"):
            tensor_weights = {k: torch.as_tensor(v) for k, v in payload["initial_weights"].items()}
            try:
                set_peft_model_state_dict(model, tensor_weights)
            except Exception as e:
                if logger:
                    logger({"msg": f"set_peft_model_state_dict failed: {e}"})

        # Load JSONL listing audio paths + transcripts
        examples = []
        with open(os.path.join(data_root, "data.jsonl")) as f:
            for line in f:
                examples.append(json.loads(line))
        dataset = Dataset.from_list(examples)

        bs = min(int(env["batch_size"]), max_batch_size)
        lr = min(float(env.get("learning_rate", max_lr)), max_lr)

        weights = train(
            model=model,
            processor=processor,
            dataset=dataset,
            data_root=data_root,
            out_dir=out_dir,
            epochs=int(env["local_epochs"]),
            batch_size=bs,
            lr=lr,
        )
        return {"weights": weights, "samples": len(dataset)}

    return fl_train_model
