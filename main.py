from yolo.model import YOLO

if __name__ == '__main__':
    model = YOLO("yolov5s.yaml", task='detect')
    model.train(data="./yolo/cfg/datasets/coco.yaml",
                device='0', imgsz=640, epochs=100, batch=16,
                box=0.05, cls=0.5, obj=1)
