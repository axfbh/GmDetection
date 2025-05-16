from typing import Any, OrderedDict
from numbers import Number

from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

import torch
from tqdm import tqdm
from torch import Tensor


def format_coco_metric(metric_name, iou_range, area, max_dets, value):
    """生成严格对齐的 COCO 风格指标输出"""
    # 固定各参数部分的宽度（根据 COCO 官方输出调整）
    iou_str = f"IoU={iou_range}".ljust(13)  # 对齐 "IoU=0.50:0.95" 等
    area_str = f"area={area:>6}".ljust(10)  # 对齐 "area=all"
    maxdets_str = f"maxDets={max_dets:>3}".ljust(11)  # 对齐 "maxDets=100"

    # 组合参数部分并控制总宽度
    params = f"{iou_str} | {area_str} | {maxdets_str}"

    # 指标名称固定宽度（对齐 AP/AR 类型）
    metric_str = f"{metric_name}".ljust(23)  # 如 "Average Precision  (AP)"

    # 数值部分右对齐，固定 7 字符宽度（含小数点后3位）
    return f"{metric_str} @[ {params} ] = {value:>5.3f}"


def print_coco_report(results):
    results = results.copy()
    for k, v in results.items():
        if isinstance(v, Tensor) and v.dtype == torch.float:
            v = v.item()
            results[k] = 0 if v < 0 else v

    report_metrics = [
        ("Average Precision  (AP)", "0.50:0.95", "all", 100, results["map"]),
        ("Average Precision  (AP)", "0.50", "all", 100, results["map_50"]),
        ("Average Precision  (AP)", "0.75", "all", 100, results["map_75"]),
        ("Average Precision  (AP)", "0.50:0.95", "small", 100, results["map_small"]),
        ("Average Precision  (AP)", "0.50:0.95", "medium", 100, results["map_medium"]),
        ("Average Precision  (AP)", "0.50:0.95", "large", 100, results["map_large"]),
        ("Average Recall     (AR)", "0.50:0.95", "all", 1, results["mar_1"]),
        ("Average Recall     (AR)", "0.50:0.95", "all", 10, results["mar_10"]),
        ("Average Recall     (AR)", "0.50:0.95", "all", 100, results["mar_100"]),
        ("Average Recall     (AR)", "0.50:0.95", "small", 100, results["mar_small"]),
        ("Average Recall     (AR)", "0.50:0.95", "medium", 100, results["mar_medium"]),
        ("Average Recall     (AR)", "0.50:0.95", "large", 100, results["mar_large"]),
    ]

    print()
    for metric in report_metrics:
        print(format_coco_metric(*metric))
    print()


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

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        results = pl_module.coco_evaluator.compute()

        fitness = 0.9 * results['map'].item() + 0.1 * results['map_50'].item()

        pl_module.log('fitness', fitness)

        print_coco_report(results)

