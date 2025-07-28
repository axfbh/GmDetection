from gmdet import DETR

if __name__ == '__main__':
    model = DETR("detr.yaml", task='detect')
    model.train(data="coco.yaml", device='0', imgsz=640, epochs=400, batch=8, box=5, cls=1, giou=2,
                eos_coef=0.1,
                optimizer='AdamW',
                lr0=0.0001,
                lrb=0.00001,
                momentum=0.9,
                warmup_bias_lr=0.0001,
                warmup_weight_lr=0.0001,
                warmup_weight_backbone_lr=0.00001,
                warmup_momentum=0.88,
                lrf=1)

    # model = DETR("./runs/detect/train/version_58/checkpoints/best.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.7, max_det=300)
