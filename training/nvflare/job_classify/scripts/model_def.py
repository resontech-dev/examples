from timm_utils import make_fl_adapter

# Vision Transformer fine-tuned on Food-101 by `nateraw`.
# Published metrics on the HF model card:
#   accuracy: 0.8913 | val_loss: 0.4501 | trained 5 epochs, batch 128, LR 2e-4
# https://huggingface.co/nateraw/food
PRETRAINED_ID = "vit_base_patch16_224.augreg_in21k_ft_in1k"
NUM_CLASSES = None    # None = inferred from shard's data.yaml (Food-101 → 101)
IMG_SIZE = 224

yolo_model, fl_train_model = make_fl_adapter(PRETRAINED_ID, num_classes=NUM_CLASSES, imgsz=IMG_SIZE)
