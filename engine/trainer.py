import os
from typing import Any, Dict, Mapping
from omegaconf import OmegaConf

import lightning as L
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from utils.lightning_utils import LitProgressBar

import torch

from engine.utils import ip_load
from utils.torch_utils import smart_optimizer, ModelEMA, smart_distribute, select_device


class BaseTrainer(LightningModule):
    def __init__(self, cfg):
        super(BaseTrainer, self).__init__()

        self.ema = None

        self.loss_names = None

        self.data = None
        self.train_set = None
        self.train_dataset = None
        self.train_loader = None
        self.val_set = None
        self.nc = None

        self.lr_lambda = None
        self.lightning_trainer = None
        self.coco_evaluator = None

        self.args = cfg
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs

        self.save_hyperparameters(self.args)

    def _setup_trainer(self):
        self.model.train()
        self.model.requires_grad_(True)

        device = select_device(self.args.device, self.batch_size)

        accelerator = self.args.device if self.args.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='box_loss',
                                              mode='min',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)

        checkpoint_callback.FILE_EXTENSION = '.pt'

        progress_bar_callback = LitProgressBar(10)

        self.lightning_trainer = L.Trainer(
            accelerator=accelerator,
            devices=device,
            num_nodes=self.args.num_nodes,
            logger=TensorBoardLogger(save_dir=f'./{self.args.project}/{self.args.task}', name=self.args.name),
            strategy=smart_distribute(self.args.num_nodes, self.device, ip_load(), "8888", "0"),
            max_epochs=self.args.epochs,
            accumulate_grad_batches=max(round(self.args.nbs / self.batch_size), 1),
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, progress_bar_callback]
        )

    def fit(self):
        self._setup_trainer()
        self.train_set, self.val_set = self.get_dataset()
        self.lightning_trainer.fit(self, ckpt_path=self.args.model if self.args.resume else None)

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

        self.lr_lambda = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      last_epoch=self.current_epoch - 1,
                                                      lr_lambda=self.lr_lambda)

        return [optimizer], [scheduler]

    def configure_model(self) -> None:
        self.model.device = self.device
        self.model.args = self.args

    def on_train_start(self) -> None:
        self.ema = ModelEMA(self.model, updates=self.args.updates)

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

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        loss, loss_items = self(batch)

        loss_dict = {name: ls for name, ls in zip(self.loss_names, loss_items)}

        self.log_dict(loss_dict,
                      on_step=True,
                      on_epoch=True,
                      sync_dist=True,
                      prog_bar=True,
                      batch_size=self.batch_size)

        return loss * self.trainer.accumulate_grad_batches * self.trainer.world_size

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_closure=None,
    ) -> None:
        super(BaseTrainer, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['ema'] = self.ema.ema
        checkpoint['updates'] = self.ema.updates
        checkpoint['state_dict'] = self.ema.ema.state_dict()

    def load_state_dict(self, *args, **kwargs):
        """
           模型参数，改在外部加载
        """
        pass
