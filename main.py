from models import YOLO
from models import DETR

# 还没做的
# 更新 dataset 格式 兼容 map
# 更新 map print 方式
# 更新 log 输出
if __name__ == '__main__':
    model = YOLO("yolov3s.yaml", task='detect')
    model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)

    # model = YOLO("yolov5s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)

    # model = YOLO("yolov8s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=7.5, cls=0.5, dfl=1.5)

    # model = YOLO("./runs/detect/train/version_0/checkpoints/last.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.7, max_det=300)

    # model = DETR(r"./runs/detect/train/version_3/checkpoints/last.pt", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=400, batch=24, resume=True)
