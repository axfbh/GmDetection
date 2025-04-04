from typing import Any

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from engine.trainer import BaseTrainer
from dataset.coco_dataset import build_coco_dataset, build_dataloader
from yolo.ops.nmsv3_7 import non_max_suppression


class DetectionTrainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_names = "box_loss", "obj_loss", "cls_loss"

    def build_dataset(self, img_path, ann_path, mode="train"):
        return build_coco_dataset(img_path, ann_path, self.args.imgsz, mode)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = self.build_dataset(self.train_set['image'], self.train_set['ann'], "train")
        self.train_loader = build_dataloader(dataset, self.batch_size, self.args.workers, True, True)
        return self.train_loader

    def val_dataloader(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        dataset = self.build_dataset(self.val_set['image'], self.val_set['ann'], "val")
        self.val_loader = build_dataloader(dataset, self.batch_size, self.args.workers, False, True)
        return self.val_loader

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        preds = non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            max_det=self.args.max_det,
            labels=[],
            multi_label=False,
            agnostic=False,
        )

        return [{'scores': p[:, 4], 'labels': p[:, 5] + 1, 'boxes': p[:, :4]} for p in preds]
