from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT

from gmdet.engine.validator import BaseValidator
from gmdet.data.dataset import build_yolo_dataset, build_dataloader
from gmdet.data.ops import NestedTensor


class SegmentationValidator(BaseValidator):

    def build_dataset(self, img_path, mode):
        return build_yolo_dataset(img_path, self.args.imgsz, self.data, self.args.task, mode)

    def setup(self, stage: str):
        self.val_dataset = self.build_dataset(self.val_set, self.args.mode)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self.val_loader = build_dataloader(self.val_dataset,
                                           batch=self.batch_size * 2,
                                           workers=self.args.workers,
                                           shuffle=False,
                                           persistent_workers=True)
        return self.val_loader

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        将 dataloader 的 collate_fn 放在这里
        :param batch:
        :param dataloader_idx:
        :return:
        """
        images = batch[0]
        targets = batch[1]

        dtype = images[0].dtype
        device = images[0].device
        channel, _, _ = images[0].shape
        b = len(images)

        batch_shape = [b, channel, self.args.imgsz, self.args.imgsz]
        pad_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        for i, (img, tg, pad_tensor) in enumerate(zip(images, targets, pad_tensors)):
            m = tg['masks']
            pad_m = torch.zeros((len(m), self.args.imgsz, self.args.imgsz), dtype=torch.long, device=device)
            c, h, w = img.shape
            pad_tensor[: c, : h, : w].copy_(img)
            pad_m[:, :h, :w].copy_(m)
            targets[i]['masks'] = pad_m

        batch[0] = pad_tensors
        return batch
