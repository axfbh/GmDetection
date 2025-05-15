from models import YOLO
from models import DETR
import torch
from torchmetrics.detection import MeanAveragePrecision

if __name__ == '__main__':
    model = YOLO("yolov5s.yaml", task='detect')
    model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)

    # model = YOLO("yolov8s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=16, box=7.5, cls=0.5, dfl=1.5)

    # model = YOLO("./runs/detect/train/version_4/checkpoints/last.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.5, max_det=300)

    # model = DETR(r"./runs/detect/train/version_3/checkpoints/last.pt", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=400, batch=24, resume=True)
