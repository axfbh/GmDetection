from engine.model import Model
from models.detr.detect.train import DetectionTrainer
from models.detr.detect.val import DetectionValidator
from models.detr.modules import Detr


class DETR(Model):
    def __init__(self, model="detr.pt", task=None):
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": {
                    'detr': Detr,
                },
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
            },
        }
