from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT

from gmdet.engine.validator import BaseValidator
from gmdet.models.yolo.utils import nms
from gmdet.data.dataset import build_yolo_dataset, build_dataloader


class DetectionValidator(BaseValidator):

    def build_dataset(self, img_path):
        return build_yolo_dataset(img_path, self.args.imgsz, self.data, self.args.task, self.args.mode)

    def setup(self, stage: str) -> None:
        self.val_dataset = self.build_dataset(self.val_set)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self.val_loader = build_dataloader(self.val_dataset, self.batch_size * 2,
                                           workers=self.args.workers,
                                           shuffle=False,
                                           persistent_workers=True)
        return self.val_loader

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""

        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            max_det=self.args.max_det,
            labels=[],
            multi_label=False,
            agnostic=False,
        )

        return [{'boxes': p[:, :4], 'scores': p[:, 4], 'labels': p[:, 5].long()} for p in preds]

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        将 dataloader 的 collate_fn 放在这里
        :param batch:
        :param dataloader_idx:
        :return:
        """
        images = batch[0]

        dtype = images[0].dtype
        device = images[0].device
        c, _, _ = images[0].shape
        batch_shape = [len(images), c, self.args.imgsz, self.args.imgsz]
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for i, (img, pad_img) in enumerate(zip(images, tensor)):
            c, h, w = img.shape
            pad_img[: c, : h, : w].copy_(img)

        batch[0] = tensor
        return batch
