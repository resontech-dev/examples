from pythia_utils import make_fl_adapter

# Pythia-1B from-scratch recipe (verbatim from EleutherAI/pythia/blob/main/models/1B/pythia-1B.yml):
#   GPTNeoX 1B params, vocab 50304, seq 2048
#   Adam (β1=0.9, β2=0.95, eps=1e-8), weight_decay=0.1, max_grad_norm=1.0
#   LR=6e-4 cosine to 6e-5, 1% warmup, 143k steps total
#   Batch 2M tokens (= 1024 sequences × 2048)
#   bf16 mixed, gradient_checkpointing, FSDP shard
# Trained on The Pile (300B tokens, 1 epoch); 32× A100-40GB × ~1 day.
# Published metrics: see https://github.com/EleutherAI/pythia/tree/main/results
#
# Smaller variants (faster to train from scratch):
#   "pythia-70m" — 70M params, ~50 GPU-hr to retrain from scratch
#   "pythia-160m" — 160M params, ~190 GPU-hr
#   "pythia-410m" — 410M params, ~600 GPU-hr
#   "pythia-1b" — 1B params, ~1300 GPU-hr (default)
CONFIG_NAME = "pythia-1b"
MAX_SEQ_LENGTH = 2048
MAX_BATCH_SIZE = 4

yolo_model, fl_train_model = make_fl_adapter(
    CONFIG_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    max_batch_size=MAX_BATCH_SIZE,
)
