from engine.model import Model
from yolo.detect.train import DetectionTrainer
from yolo.models import YoloV4, YoloV5
from functools import partial


class YOLO(Model):
    def __init__(self, model="yolo11n.pt", task=None):
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": {
                    'yolov3': YoloV4,
                    'yolov4': YoloV4,
                    'yolov5': YoloV5,
                    # 'v7': YoloV7(anchors=anchors, num_classes=nc, scales=scales),
                    # 'v8': YoloV8(num_classes=nc + 1, scales=scales),
                },
                "trainer": DetectionTrainer,
            },
        }
