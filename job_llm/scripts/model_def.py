from llm_utils import make_fl_adapter

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
LORA_RANK = 16
MAX_LENGTH = 512

fl_train_model = make_fl_adapter(BASE_MODEL, lora_rank=LORA_RANK, max_length=MAX_LENGTH)
