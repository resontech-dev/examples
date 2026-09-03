from whisper_utils import make_fl_adapter

# Canonical Whisper LoRA recipe (HF blog `fine-tune-whisper` + PEFT examples):
#   r=32, alpha=64, dropout=0.05, target_modules=["q_proj", "v_proj"]
#   adamw_torch, LR=1e-3 (LoRA is higher than full-FT 1e-5), warmup=50 steps
#   batch=8, fp16, gradient_checkpointing, 3 epochs (or max_steps=5000)
# Sources:
#   https://huggingface.co/blog/fine-tune-whisper
#   https://github.com/Vaibhavs10/fast-whisper-finetuning
BASE_MODEL = "openai/whisper-small"   # tiny / base / small / medium / large-v3
LANGUAGE = "en"
LORA_RANK = 32

fl_train_model = make_fl_adapter(BASE_MODEL, language=LANGUAGE, lora_rank=LORA_RANK)
