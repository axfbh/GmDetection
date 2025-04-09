from engine.model import Model
from models.yolo.detect.train import DetectionTrainer
from models.yolo.detect.val import DetectionValidator
from models.yolo.modules import YoloV4, YoloV5, YoloV7


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
                    'yolov7': YoloV7,
                    # 'v8': YoloV8(num_classes=nc + 1, scales=scales),
                },
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
            },
        }
