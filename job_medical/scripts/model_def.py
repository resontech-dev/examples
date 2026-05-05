from monai_utils import make_fl_adapter

# HAM10000 — 7-class dermatoscopic skin lesion classification (a real medical task,
# not a toy MNIST-equivalent). Dataset published as `marmal88/skin_cancer` on HF
# (13,354 images split 9,580/2,490/1,294 train/val/test).
# Note: no popular pretrained model has published HAM10000 metrics on its HF card,
# so your FL run produces the reference training data.
NUM_CLASSES = 7   # akiec, bcc, bkl, df, mel, nv, vasc

yolo_model, fl_train_model = make_fl_adapter(num_classes=NUM_CLASSES)
