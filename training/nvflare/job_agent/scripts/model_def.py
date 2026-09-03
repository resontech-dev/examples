from llm_utils import make_fl_adapter

# meta-llama/Llama-3.1-8B-Instruct — 8B params, 128K context, agentic-tuned.
# Published benchmarks on the HF model card (verbatim):
#   MMLU (5-shot): 69.4  | MMLU-Pro (5-shot CoT): 48.3 | IFEval: 80.4
#   GSM-8K (8-shot CoT): 84.5 | MATH (0-shot CoT): 51.9
#   HumanEval (0-shot pass@1): 72.6 | MBPP++ (0-shot pass@1): 72.8
#   Tool use — BFCL (0-shot): 76.1 | API-Bank (0-shot): 82.6
# Trained on 15T+ tokens. Weights gated; request access on HF or use the
# Unsloth 4-bit quant `unsloth/llama-3.1-8b-Instruct-bnb-4bit` (no auth).
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
BASE_MODEL = "unsloth/llama-3.1-8b-Instruct-bnb-4bit"
LORA_RANK = 16
MAX_SEQ_LENGTH = 2048
MAX_BATCH_SIZE = 2          # 8B model needs more memory than 7B; cap at 2

fl_train_model = make_fl_adapter(
    BASE_MODEL,
    lora_rank=LORA_RANK,
    max_seq_length=MAX_SEQ_LENGTH,
    max_batch_size=MAX_BATCH_SIZE,
)
