from gmdet.engine.model import Model
from gmdet.models.liteseg.segment import SegmentationTrainer, SegmentationValidator
from gmdet.models.liteseg.modules import LiteSeg


class LITESEG(Model):
    def __init__(self, model="liteseg.pt", task=None):
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "segment": {
                "model": {
                    'liteseg': LiteSeg,
                },
                "trainer": SegmentationTrainer,
                "validator": SegmentationValidator,
            },
        }
