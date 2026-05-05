from embed_utils import make_fl_adapter

# BAAI/bge-base-en-v1.5 — 110M params, 768-dim embeddings, seq len 512.
# Published MTEB metrics on the HF card (verbatim):
#   MTEB Average (56 datasets): 63.55
#   Retrieval (15): 53.25 | STS (10): 82.40 | Pair Classification (3): 86.55
#   Classification (12): 75.53 | Clustering (11): 45.77 | Reranking (4): 58.86
# https://huggingface.co/BAAI/bge-base-en-v1.5
BASE_MODEL = "BAAI/bge-base-en-v1.5"

yolo_model, fl_train_model = make_fl_adapter(BASE_MODEL)
