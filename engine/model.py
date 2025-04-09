from torch import nn
from pathlib import Path
from omegaconf import OmegaConf

from engine.utils import yaml_model_load, attempt_load_one_weight


class Model:
    def __init__(self, model, task=None) -> None:
        super().__init__()
        self.trainer = None
        self.overrides = {}

        model = str(model).strip()

        if Path(model).suffix in {".yaml", ".yml"}:
            assert task is not None, 'yaml文件加载模型，必须填写task参数'
            self._new(model, task=task)
        else:
            self._load(model)

    def _new(self, cfg: str, task) -> None:
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task
        self.model = self._smart_load('model')[cfg_dict['model']](cfg_dict)

        self.overrides["model"] = self.cfg
        self.overrides["updates"] = 0
        self.overrides["task"] = self.task

    def _load(self, weights: str) -> None:
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.task = self.model.args.task

        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.overrides["model"] = weights
        self.overrides["updates"] = self.ckpt.get('updates', 0)
        self.overrides["task"] = self.task
        self.model_name = weights

    def train(
            self,
            *,
            data,
            num_nodes=1,
            **kwargs,
    ):
        args = OmegaConf.create({**self.overrides, **OmegaConf.load('./cfg/default.yaml'), "mode": 'train'})
        args.update(**kwargs)
        args.update({'data': data, 'num_nodes': num_nodes})

        trainer = self._smart_load("trainer")(args)
        trainer.add_module('model', self.model)
        trainer.fit()

    def val(self, *, data, **kwargs):
        args = OmegaConf.create({**self.overrides, "mode": 'val'})
        args.update(**kwargs)
        args.update({'data': data})

        validator = self._smart_load("validator")(args)
        validator.add_module('model', self.model)
        validator.validate()

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        include = {"imgsz", "data", "task", "single_cls"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        return self.task_map[self.task][key]

    @property
    def task_map(self) -> dict:
        raise NotImplementedError("Please provide task map for your model!")
