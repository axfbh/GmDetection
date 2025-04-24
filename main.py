from models import YOLO
from models import DETR

if __name__ == '__main__':
    # model = YOLO("yolov8s.yaml", task='detect')
    # model = YOLO("yolov5s.yaml", task='detect')
    # model = YOLO("./runs/detect/train/version_29/checkpoints/last.pt", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=16, box=7.5, cls=0.5, dfl=1.5)

    # model = YOLO("./runs/detect/train/version_17/checkpoints/last.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.7, max_det=300)

    model = DETR("detr.yaml", task='detect')
    model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=6,
                weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2})
