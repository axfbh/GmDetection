from gmdet import YOLO

if __name__ == '__main__':
    # model = YOLO("yolov3s.yaml", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1,
    #             warmup_weight_lr=0,
    #             warmup_weight_backbone_lr=0)

    model = YOLO("yolov5s.yaml", task='detect')
    model.train(data="./tests/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=0.05, cls=0.5, obj=1,
                warmup_weight_lr=0,
                warmup_weight_backbone_lr=0)

    # model = YOLO("yolov8s.yaml", task='detect')
    # model.train(data="./tests/coco.yaml", device='0', imgsz=640, epochs=100, batch=8, box=7.5, cls=0.5, dfl=1.5,
    #             warmup_weight_lr=0,
    #             warmup_weight_backbone_lr=0)

    # model = YOLO("./runs/detect/train/version_0/checkpoints/last.pt", task='detect')
    # model.train(data="./cfg/datasets/coco.yaml", device='0', resume=True, imgsz=640, epochs=100, batch=8, box=7.5,
    #             cls=0.5, dfl=1.5)
