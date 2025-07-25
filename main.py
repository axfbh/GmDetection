from gmdet import YOLO
from gmdet import DETR
from gmdet import LITESEG
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    # model = YOLO("yolov3s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)

    # model = YOLO("yolov5s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)

    # model = YOLO("yolov8s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=7.5, cls=0.5, dfl=1.5)

    # model = YOLO("./runs/detect/train/version_0/checkpoints/last.pt", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', resume=True, imgsz=640, epochs=100, batch=8, box=7.5,
    #             cls=0.5, dfl=1.5)

    # model = YOLO("./runs/detect/train/version_58/checkpoints/best.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.7, max_det=300)

    model = DETR("detr.yaml", task='detect')
    model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=400, batch=8,
                optimizer='AdamW',
                lr0=0.0001,
                lrb=0.00001,
                warmup_bias_lr=0.001,
                warmup_weight_lr=0.001,
                warmup_weight_backbone_lr=0.001,
                momentum=0.9,
                warmup_momentum=0.88,
                lrf=1)

    # model = LITESEG('liteseg.yaml', task='segment')
    # model.train(data="./cfg/datasets/coco-seg.yaml", device='0', imgsz=640, epochs=400, batch=8)
