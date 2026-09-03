from imagenet_utils import make_fl_adapter

# timm A2 recipe (from `ResNet Strikes Back`, https://arxiv.org/abs/2110.00476):
#   ResNet-50 random init, LAMB lr=5e-3, weight_decay=0.02
#   Cosine schedule + 5-epoch warmup, BCE loss + label smoothing 0.1
#   Mixup α=0.1, CutMix α=1.0 (switch_prob=0.5)
#   Stochastic depth 0.05, RandAugment(2, 7), repeated augmentation
#   Batch 2048 across GPUs, 300 epochs, fp16 AMP
# Published: top-1 79.8% on ImageNet val (timm/resnet50.a2_in1k)
MODEL_NAME = "resnet50"   # timm name; "resnet50.a2_in1k" reproduces verbatim
IMG_SIZE = 224
LR = 5e-3
WEIGHT_DECAY = 0.02
DROP_PATH_RATE = 0.05

yolo_model, fl_train_model = make_fl_adapter(
    MODEL_NAME,
    imgsz=IMG_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    drop_path_rate=DROP_PATH_RATE,
)
