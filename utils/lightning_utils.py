import sys
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm, convert_inf
from typing import Any, OrderedDict
from numbers import Number

import torch
from torchvision.utils import make_grid
from tqdm import tqdm


class LitTqdm(Tqdm):
    def format_meter(*args, **kwargs):
        item = tqdm.format_meter(**kwargs)
        item = item.replace(',', ' ')
        item = item.replace('=', ': ')
        return item


class LitProgressBar(TQDMProgressBar):
    BAR_FORMAT = "{desc} [{n_fmt}/{total_fmt}]{postfix}  剩余:{remaining}"

    def init_train_tqdm(self) -> LitTqdm:
        """Override this to customize the tqdm bar for training."""
        return LitTqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def on_train_epoch_start(self, trainer, *_: Any) -> None:
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch [{trainer.current_epoch}]")

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

# class LitProgressBar(ProgressBar):
#
#     def __init__(self, refresh_rate=1):
#         super().__init__()  # don't forget this :)
#         self.enable = True
#         self.refresh_rate = refresh_rate
#
#     def disable(self):
#         self.enable = False
#
#     def get_metrics(self, trainer, model):
#         items = super().get_metrics(trainer, model)
#         items.pop("v_num", None)
#         return items
#
#     def get_meters(self, trainer, pl_module):
#         meters = self.get_metrics(trainer, pl_module)
#
#         meters_mean = self.get_meters_mean(trainer)
#
#         meters.pop("v_num", None)
#
#         loss_str = []
#
#         for name, meter in meters.items():
#             if name.endswith('step'):
#                 name = name[:-5]
#                 if name.startswith('loss') or name.endswith('loss') and len(meters_mean) > 0:
#                     loss_str.append(
#                         "{}: {:.4f} ({:.4f})".format(name, meter, meters_mean[name])
#                     )
#                 else:
#                     loss_str.append(
#                         "{}: {:.4f} ".format(name, meter)
#                     )
#
#         return '  '.join(loss_str)
#
#     def get_meters_mean(self, trainer) -> dict:
#         meters_name = {}
#         for result in trainer._results.result_metrics:
#             name = result.meta.name
#             value = result.value
#             cumulated_batch_size = result.cumulated_batch_size
#             meters_name[name] = (value / cumulated_batch_size).item()
#         return meters_name
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
#
#         if batch_idx % self.refresh_rate == 0 or (batch_idx == self.total_train_batches - 1):
#             MB = 1024.0 * 1024.0 * 1024.0
#
#             delimiter = '  '
#
#             space_fmt = ':' + str(len(str(batch_idx))) + 'd'
#
#             log_msg = delimiter.join([
#                 'Epoch: [{}]'.format(trainer.current_epoch),
#                 '[{0' + space_fmt + '}/{1}]',
#                 '{meters}',
#                 'max mem: {memory:.2f} GB'
#             ])
#
#             print(log_msg.format(batch_idx,
#                                  self.total_train_batches,
#                                  meters=self.get_meters(trainer, pl_module),
#                                  memory=torch.cuda.max_memory_allocated() / MB))
#
#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
#         super(LitProgressBar, self).on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx,
#                                                             dataloader_idx)
#         if batch_idx % self.refresh_rate == 0 or (batch_idx == self.total_val_batches - 1):
#             MB = 1024.0 * 1024.0 * 1024.0
#
#             delimiter = '  '
#
#             space_fmt = ':' + str(len(str(batch_idx))) + 'd'
#
#             log_msg = delimiter.join([
#                 'Val: [{}]'.format(trainer.current_epoch),
#                 '[{0' + space_fmt + '}/{1}]',
#                 '{meters}',
#                 'max mem: {memory:.2f} GB'
#             ])
#
#             print(log_msg.format(batch_idx,
#                                  self.total_val_batches,
#                                  meters=self.get_meters(trainer, pl_module),
#                                  memory=torch.cuda.max_memory_allocated() / MB))
