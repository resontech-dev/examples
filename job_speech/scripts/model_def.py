from whisper_utils import make_fl_adapter

BASE_MODEL = "openai/whisper-small"   # whisper-tiny / base / small / medium / large-v3
LANGUAGE = "en"
LORA_RANK = 16

fl_train_model = make_fl_adapter(BASE_MODEL, language=LANGUAGE, lora_rank=LORA_RANK)
