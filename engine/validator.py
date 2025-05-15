from omegaconf import OmegaConf

import lightning as L
from lightning import LightningModule

from utils.lightning_utils import LitProgressBar
from utils.coco_eval import CocoEvaluator
from dataset.coco_dataset import get_coco_api_from_dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops.boxes import box_convert


class BaseValidator(LightningModule):
    def __init__(self, args=None):
        super(BaseValidator, self).__init__()
        self.args = args

        self.coco_evaluator = None

        self.data = None
        self.val_set = None
        self.val_dataset = None
        self.val_loader = None

        self.lightning_validator = None

        self.iou_types = {
            'detect': 'bbox',
            'segment': "seg"
        }

        self.batch_size = None if self.args is None else int(self.args.batch * 0.5)

    def _setup_validator(self):
        self.model.eval()

        device = [0]

        accelerator = self.args.device if self.args.device in ["cpu", "tpu", "ipu", "hpu", "mps"] else 'gpu'

        progress_bar_callback = LitProgressBar(10)

        self.lightning_validator = L.Trainer(
            accelerator=accelerator,
            devices=device,
            num_nodes=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            callbacks=[progress_bar_callback]
        )

    def validate(self):
        self._setup_validator()
        self.val_set = self.get_dataset()
        self.lightning_validator.validate(self)

    def get_dataset(self):
        self.data = OmegaConf.load(self.args.data)
        return self.data['val']

    def configure_model(self) -> None:
        self.model.device = self.device
        self.model.args = self.args

    def on_validation_start(self):
        base_ds = get_coco_api_from_dataset(self.val_loader.dataset)

        self.coco_evaluator = CocoEvaluator(base_ds, [self.iou_types[self.args.task]])
        # self.coco_evaluator = MeanAveragePrecision()

    def forward(self, batch):
        return self.model(batch)

    def validation_step(self, batch, batch_idx):
        targets = batch[2]

        preds = self.ema.ema(batch) if hasattr(self, 'ema') else self(batch)
        preds = self.postprocess(preds)
        # true_bboxs = []
        # for target in targets:
        #     true_bboxs.append(
        #         {
        #             'boxes': box_convert(target['boxes'], 'cxcywh', 'xyxy') * self.args.imgsz,
        #             'labels': target['labels'],
        #         }
        #
        #     )
        res = {target['image_id'].item(): output for target, output in zip(targets, preds)}
        self.coco_evaluator.update(res)
        # self.coco_evaluator.update(preds, true_bboxs)

    def on_validation_epoch_end(self) -> None:
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        # results = self.coco_evaluator.compute()
        # print(results)

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds
