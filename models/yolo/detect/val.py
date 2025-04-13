from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT

from engine.validator import BaseValidator
from models.yolo.utils import nmsv3_7, nmsv8_11
from dataset.coco_dataset import build_coco_dataset, build_dataloader


class DetectionValidator(BaseValidator):

    def build_dataset(self, img_path, ann_path, mode="val"):
        return build_coco_dataset(img_path, ann_path, self.args.imgsz, mode)

    def prepare_data(self):
        model = self.ema.ema if hasattr(self, 'ema') else self.model

        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if model.__class__.__name__ in ['YoloV8',
                                                                                             'YoloV11'] else "box_loss", "obj_loss", "cls_loss"
        self.val_dataset = self.build_dataset(self.val_set['image'], self.val_set['ann'], "val")

    def val_dataloader(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self.val_loader = build_dataloader(self.val_dataset, self.batch_size * 2,
                                           workers=self.args.workers,
                                           shuffle=False,
                                           persistent_workers=True)
        return self.val_loader

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""

        model = self.ema.ema if hasattr(self, 'ema') else self.model

        g = 0 if model.__class__.__name__ in ['YoloV8', 'YoloV11'] else 1

        non_max_suppression = nmsv3_7.non_max_suppression if g else nmsv8_11.non_max_suppression

        preds = non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            max_det=self.args.max_det,
            labels=[],
            multi_label=False,
            agnostic=False,
        )

        return [{'scores': p[:, 4], 'labels': p[:, 5] + g, 'boxes': p[:, :4]} for p in preds]

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        将 dataloader 的 collate_fn 放在这里
        :param batch:
        :param dataloader_idx:
        :return:
        """
        images = batch[0]
        self.model.orig_size = torch.zeros((len(images), 2))

        dtype = images[0].dtype
        device = images[0].device
        c, _, _ = images[0].shape
        batch_shape = [len(images), c, self.args.imgsz, self.args.imgsz]
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for i, (img, pad_img) in enumerate(zip(images, tensor)):
            c, h, w = img.shape
            pad_img[: c, : h, : w].copy_(img)
            self.model.orig_size[i, 0] = h
            self.model.orig_size[i, 1] = w

        batch[0] = tensor
        return batch
