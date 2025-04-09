from lightning.pytorch.callbacks import ProgressBar
import torch


class LitProgressBar(ProgressBar):

    def __init__(self, refresh_rate=1):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.refresh_rate = refresh_rate
        self.prev_mean = None

    def disable(self):
        self.enable = False

    def get_meters(self, trainer, pl_module):
        meters = self.get_metrics(trainer, pl_module)

        meters_mean = self.get_meters_mean(trainer)

        meters.pop('v_num')

        loss_str = []

        for name, meter in meters.items():
            if name.endswith('step'):
                name = name[:-5]
                if name.startswith('loss') or name.endswith('loss') and len(meters_mean) > 0:
                    if self.prev_mean is None:
                        loss_str.append(
                            "{}: {:.4f} ({:.4f})".format(name, meter, meters_mean[name])
                        )
                    else:
                        loss_str.append(
                            "{}: {:.4f} ({:.4f}-{:.4f})".format(name, meter, self.prev_mean[name], meters_mean[name])
                        )
                else:
                    loss_str.append(
                        "{}: {:.4f} ".format(name, meter)
                    )

        return '  '.join(loss_str)

    def get_meters_mean(self, trainer) -> dict:
        meters_name = {}
        for result in trainer._results.result_metrics:
            name = result.meta.name
            value = result.value
            cumulated_batch_size = result.cumulated_batch_size
            meters_name[name] = (value / cumulated_batch_size).item()
        return meters_name

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)

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
                                 meters=self.get_meters(trainer, pl_module),
                                 memory=torch.cuda.max_memory_allocated() / MB))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.prev_mean = self.get_meters_mean(trainer)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.prev_mean = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super(LitProgressBar, self).on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx,
                                                            dataloader_idx)
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
                                 meters=self.get_meters(trainer, pl_module),
                                 memory=torch.cuda.max_memory_allocated() / MB))
