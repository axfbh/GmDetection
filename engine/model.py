from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from engine.utils import yaml_model_load, ip_load
from tools.torch_utils import select_device, smart_distribute
from tools.lightning_utils import LitProgressBar


class Model:
    def __init__(self, model, task=None) -> None:
        super().__init__()
        self.trainer = None
        self.overrides = {}

        model = str(model).strip()

        if Path(model).suffix in {".yaml", ".yml"}:
            assert task is not None, 'yaml文件加载模型，必须填写task参数'
            self._new(model, task=task)

    def _new(self, cfg: str, task) -> None:
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task
        self.model = self._smart_load('model')[cfg_dict['model']](cfg_dict)

        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

    def train(
            self,
            *,
            data,
            num_nodes=1,
            **kwargs,
    ):
        args = OmegaConf.create({**self.overrides, **OmegaConf.load('./yolo/cfg/default.yaml')})
        args.update(**kwargs)
        args.update({'data': data})

        self.trainer = self._smart_load("trainer")(args)
        self.trainer.add_module('model', self.model)

        self.device = select_device(args.device, args.batch)
        accelerator = self.device if self.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        checkpoint_callback = ModelCheckpoint(filename='best',
                                              save_last=True,
                                              monitor='box_loss',
                                              mode='max',
                                              auto_insert_metric_name=False,
                                              enable_version_counter=False)
        checkpoint_callback.FILE_EXTENSION = '.pt'

        progress_bar_callback = LitProgressBar(10)

        l_trainer = L.Trainer(
            accelerator=accelerator,
            devices=self.device,
            num_nodes=num_nodes,
            logger=TensorBoardLogger(save_dir=f'./{args.project}', name=args.name),
            strategy=smart_distribute(num_nodes, self.device, ip_load(), "8888", "0"),
            max_epochs=args.epochs,
            gradient_clip_val=10,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback, progress_bar_callback]
        )

        l_trainer.fit(self.trainer)

    def _smart_load(self, key: str):
        return self.task_map[self.task][key]

    @property
    def task_map(self) -> dict:
        raise NotImplementedError("Please provide task map for your model!")
