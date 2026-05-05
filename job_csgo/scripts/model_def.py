from yolo_utils import make_fl_adapter

PRETRAINED_ID = "keremberke/yolov5n-csgo"
IMG_SIZE = 640

yolo_model, fl_train_model = make_fl_adapter(PRETRAINED_ID, imgsz=IMG_SIZE)
