from models import YOLO
from models import DETR
import torch

if __name__ == '__main__':

    # # 生成数据
    # pred_boxes = [
    #     {
    #         'boxes': torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45],
    #                                [40, 50, 60, 70], [5, 5, 20, 20],
    #                                [25, 35, 45, 55], [30, 30, 50, 50],
    #                                [50, 60, 70, 80], [12, 18, 32, 42],
    #                                [38, 48, 58, 68], [42, 52, 62, 72]]),
    #         'scores': torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]),
    #         'labels': torch.tensor([0] * 10)  # 假设均为类别0
    #     }
    # ]
    #
    # # 真实数据（2个框）
    # true_boxes = [
    #     {
    #         'boxes': torch.tensor([[12, 22, 32, 42], [40, 50, 60, 70]]),
    #         'labels': torch.tensor([0, 0])
    #     }
    # ]
    #
    # # 更新评估器
    # map_metric.update(pred_boxes, true_boxes)
    #
    # # 计算 mAP
    # results = map_metric.compute()
    # print(results)

    model = YOLO("yolov5s.yaml", task='detect')
    model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1)

    # model = YOLO("yolov8s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=16, box=7.5, cls=0.5, dfl=1.5)

    # model = YOLO("./runs/detect/train/version_17/checkpoints/last.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.7, max_det=300)

    # model = DETR(r"./runs/detect/train/version_3/checkpoints/last.pt", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=400, batch=24, resume=True)
