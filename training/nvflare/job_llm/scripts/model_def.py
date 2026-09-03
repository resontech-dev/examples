from llm_utils import make_fl_adapter

# Per Microsoft's `sample_finetune.py` shipped with the model:
#   LR=5e-6, cosine + warmup_ratio=0.2, batch=4 × grad_accum=1, 1 epoch
#   LoRA: r=16, alpha=32, dropout=0.05, target_modules="all-linear"
#   bf16, gradient_checkpointing, max_seq_length=2048
# https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/sample_finetune.py
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LORA_RANK = 16
MAX_LENGTH = 2048

fl_train_model = make_fl_adapter(BASE_MODEL, lora_rank=LORA_RANK, max_length=MAX_LENGTH)
