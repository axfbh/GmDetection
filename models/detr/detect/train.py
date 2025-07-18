from typing import Any

import torch

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from engine.trainer import BaseTrainer

from data.dataset import build_detect_dataset, build_dataloader

from data.ops import NestedTensor

from models.detr.detect.val import DetectionValidator


# 先执行 BaseTrainer，从 BaseTrainer super 跳到执行 DetectionValidator
# 因此 DetectionValidator 创建的重复信息会被，后续执行BaseTrainer覆盖，不影响训练时候的参数
class DetectionTrainer(BaseTrainer, DetectionValidator):

    def build_dataset(self, img_path, mode="train"):
        return build_detect_dataset(img_path, self.args.imgsz, mode)

    def setup(self, stage: str):
        self.train_dataset = self.build_dataset(self.train_set, "train")
        self.nc = max(self.train_dataset.coco.cats.keys())

        if self.val_set is not None:
            self.val_dataset = self.build_dataset(self.val_set, "val")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.train_loader = build_dataloader(self.train_dataset, self.batch_size,
                                             workers=self.args.workers,
                                             shuffle=True,
                                             persistent_workers=True)
        return self.train_loader

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
        pad_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for i, (img, pad_tensor, m) in enumerate(zip(images, pad_tensors, mask)):
            c, h, w = img.shape
            pad_tensor[: c, : h, : w].copy_(img)
            m[: h, :w] = False

        batch[0] = NestedTensor(pad_tensors, mask)
        batch[1] = torch.stack(batch[1])
        return batch
