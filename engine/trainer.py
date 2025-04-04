import os
from typing import Any, Optional, Union
from omegaconf import OmegaConf

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import torch

from tools.coco_eval import CocoEvaluator
from tools.torch_utils import smart_optimizer, ModelEMA
from dataset.coco_dataset import get_coco_api_from_dataset


class BaseTrainer(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.coco_evaluator = None
        self.ema = None
        self.data = None
        self.train_loader = None
        self.val_loader = None

        self.args = cfg
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs

        self.lr_lambda = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf

        self.train_set, self.val_set = self.get_dataset()

    def get_dataset(self):
        self.data = OmegaConf.load(self.args.data)
        return self.data['train'], self.data.get('val')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * accumulate / self.args.nbs
        optimizer = smart_optimizer(self,
                                    self.args.optimizer,
                                    self.args.lr0,
                                    self.args.momentum,
                                    weight_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      last_epoch=self.current_epoch - 1,
                                                      lr_lambda=self.lr_lambda)

        return [optimizer], [scheduler]

    def configure_model(self) -> None:
        self.model.device = self.device
        self.model.args = self.args

    def on_train_start(self) -> None:
        self.ema = ModelEMA(self.model)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        epoch = self.current_epoch

        ni = batch_idx + self.batch_size * epoch
        nw = max(round(self.batch_size * self.args.warmup_epochs), 100)

        if ni <= nw:
            ratio = ni / nw
            interpolated_accumulate = 1 + (self.args.nbs / self.batch_size - 1) * ratio
            self.trainer.accumulate_grad_batches = max(1, round(interpolated_accumulate))
            for j, param_group in enumerate(self.optimizers().param_groups):
                # 学习率线性插值
                lr_start = self.args.warmup_bias_lr if j == 0 else 0.0
                lr_end = param_group["initial_lr"] * self.lr_lambda(epoch)
                param_group["lr"] = lr_start + (lr_end - lr_start) * ratio

                if "momentum" in param_group:
                    param_group["momentum"] = self.args.warmup_momentum + (
                            self.args.momentum - self.args.warmup_momentum) * ratio

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        loss, loss_items = self(batch)

        loss_dict = {name: ls for name, ls in zip(self.loss_names, loss_items)}

        self.log_dict(loss_dict,
                      on_step=True,
                      on_epoch=True,
                      sync_dist=True,
                      prog_bar=True,
                      batch_size=self.trainer.train_dataloader.batch_size)

        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.ema.update(self.model)

    def on_validation_start(self) -> None:
        base_ds = get_coco_api_from_dataset(self.val_loader.dataset)
        iou_types = {
            'detect': 'bbox',
            'segment': "seg"
        }
        self.coco_evaluator = CocoEvaluator(base_ds, [iou_types[self.args.task]])

    def validation_step(self, batch, batch_idx):
        targets = batch[1]

        preds = self.ema.ema(batch)
        preds = self.postprocess(preds)
        res = {target['image_id'].item(): output for target, output in zip(targets, preds)}
        self.coco_evaluator.update(res)

    def on_validation_epoch_end(self) -> None:
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds
