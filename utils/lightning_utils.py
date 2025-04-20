from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from typing import Any, OrderedDict
from numbers import Number

import torch
from torchvision.utils import make_grid
from tqdm import tqdm


class LitTqdm(Tqdm):
    def format_meter(*args, **kwargs):
        item = tqdm.format_meter(**kwargs)
        item = item.replace(',', ' ')
        return item

    def set_description(self, desc=None, refresh=True):
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
        self.desc = desc
        if refresh:
            self.refresh()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
        # Sort in alphabetical order to be more deterministic
        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = self.format_num(postfix[key])
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        self.postfix = '  '.join(key + ': ' + postfix[key].strip()
                                 for key in postfix.keys())
        if refresh:
            self.refresh()


class LitProgressBar(ProgressBar):
    def __init__(self, refresh_rate=1):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.refresh_rate = refresh_rate

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        metrics_step = trainer._results.metrics(True)['pbar']
        metrics_step = {k[:-5]: v for k, v in metrics_step.items()}
        metrics_epoch = trainer._results.metrics(False)['pbar']
        metrics_epoch = {k[:-6]: v for k, v in metrics_epoch.items()}

        loss_str = []

        for name, meter in metrics_step.items():
            if name.startswith('loss') or name.endswith('loss') and len(metrics_epoch) > 0:
                loss_str.append(
                    "{}: {:.4f} ({:.4f})".format(name, meter, metrics_epoch[name])
                )
            else:
                loss_str.append(
                    "{}: {:.4f} ".format(name, meter)
                )

        return '  '.join(loss_str)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.refresh_rate == 0 or (batch_idx == self.total_train_batches - 1):
            MB = 1024.0 * 1024.0 * 1024.0

            delimiter = '  '

            space_fmt = ':' + str(len(str(batch_idx))) + 'd'

            log_msg = delimiter.join([
                'Epoch: [{}]'.format(trainer.current_epoch),
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'max mem: {memory:.2f} GB'
            ])

            print(log_msg.format(batch_idx,
                                 self.total_train_batches,
                                 meters=self.get_metrics(trainer, pl_module),
                                 memory=torch.cuda.max_memory_allocated() / MB))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.refresh_rate == 0 or (batch_idx == self.total_val_batches - 1):
            MB = 1024.0 * 1024.0 * 1024.0

            delimiter = '  '

            space_fmt = ':' + str(len(str(batch_idx))) + 'd'

            log_msg = delimiter.join([
                'Val: [{}]'.format(trainer.current_epoch),
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'max mem: {memory:.2f} GB'
            ])

            print(log_msg.format(batch_idx,
                                 self.total_val_batches,
                                 meters=self.get_metrics(trainer, pl_module),
                                 memory=torch.cuda.max_memory_allocated() / MB))
