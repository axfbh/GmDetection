from typing import Any

import torch

from lightning.pytorch.utilities.types import STEP_OUTPUT

from engine.validator import BaseValidator

from data.dataset import build_detect_dataset, build_dataloader

from data.ops import NestedTensor


class DetectionValidator(BaseValidator):

    def build_dataset(self, img_path, mode="val"):
        return build_detect_dataset(img_path, self.args.imgsz, mode)

    def setup(self, stage: str):
        self.val_dataset = self.build_dataset(self.val_set, "val")

    def val_dataloader(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        self.val_loader = build_dataloader(self.val_dataset, self.batch_size * 2,
                                           workers=self.args.workers,
                                           shuffle=False,
                                           persistent_workers=True)
        return self.val_loader

    def postprocess(self, preds):
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
        c, h, w = images[0].shape
        b = len(images)
        batch_shape = [b, c, self.args.imgsz, self.args.imgsz]
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for i, (img, pad_img, m) in enumerate(zip(images, tensor, mask)):
            c, h, w = img.shape
            pad_img[: c, : h, : w].copy_(img)
            m[: h, :w] = False

        batch[0] = NestedTensor(tensor, mask)
        return batch
