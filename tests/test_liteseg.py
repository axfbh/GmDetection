from gmdet import LITESEG

if __name__ == '__main__':
    model = LITESEG('liteseg.yaml', task='segment')
    model.train(data="./cfg/datasets/coco-seg.yaml", device='0', imgsz=640, epochs=400, batch=8)

    # model = LITESEG("./runs/detect/train/version_58/checkpoints/best.pt")
    # model.val(data="./cfg/datasets/coco.yaml", workers=3, device='0', batch=16, conf=0.001, iou=0.7, max_det=300)

