from yolo_utils import make_fl_adapter

# Ultralytics/YOLO11 — official YOLO11 family. Use YOLO11m for FL (medium variant).
# Published COCO val2017 mAP@50-95 on the HF model card (verbatim):
#   YOLO11n: 39.5  | YOLO11s: 47.0  | YOLO11m: 51.5  | YOLO11l: 53.4  | YOLO11x: 54.7
# YOLO11m beats YOLOv8m (50.2 mAP) with fewer params (20.1M vs 25.9M).
# https://huggingface.co/Ultralytics/YOLO11
PRETRAINED_REPO = "Ultralytics/YOLO11"
WEIGHTS_FILE = "yolo11m.pt"      # m=20.1M params, COCO mAP 51.5
IMG_SIZE = 640

yolo_model, fl_train_model = make_fl_adapter(PRETRAINED_REPO, WEIGHTS_FILE, imgsz=IMG_SIZE)
