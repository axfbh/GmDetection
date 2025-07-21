from torch import nn

from gmdet.nn.conv import CBS
from gmdet.nn.backbone import Backbone
from gmdet.nn.head import YoloHeadV4
from gmdet.models.yolo.utils.yolo_loss import YoloLossV4To7
from gmdet.utils import LOGGER


class YoloV3(nn.Module):
    def __init__(self, cfg, nc=None):
        super(YoloV3, self).__init__()

        self.yaml = cfg

        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value

        scale = self.yaml['scale']
        scales = self.yaml['scales']
        nc = self.yaml["nc"]
        width_multiple, depth_multiple = scales[scale]

        base_channels = int(width_multiple * 32)  # 32
        base_depth = max(round(depth_multiple * 2), 1)  # 1

        self.backbone = Backbone(name='Darknet',
                                 layers_to_train=['stem',
                                                  'feature1',
                                                  'feature2',
                                                  'feature3',
                                                  'feature4'],
                                 return_interm_layers={'feature2': '0',
                                                       'feature3': '1',
                                                       'feature4': '2'},
                                 base_channels=base_channels,
                                 base_depth=base_depth)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = CBS(base_channels * 32, base_channels * 8, 1)
        self.conv_for_P4 = CBS(base_channels * 16, base_channels * 8, 1)
        self.conv_for_P3 = CBS(base_channels * 8, base_channels * 8, 1)

        self.down_sample = nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)

        self.head = YoloHeadV4([base_channels * 8, base_channels * 8, base_channels * 8],
                               self.yaml.anchors,
                               nc)

    def forward(self, batch):
        x = batch[0]

        features = self.backbone(x)

        feat1, feat2, feat3 = features['0'], features['1'], features['2']

        P5 = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4 = P5_upsample + self.conv_for_P4(feat2)

        P4_upsample = self.upsample(P4)
        P3 = P4_upsample + self.conv_for_P3(feat1)

        P3_downsample = self.down_sample(P3)
        P4 = P3_downsample + P4

        P4_downsample = self.down_sample(P4)
        P5 = P4_downsample + P5

        # ----------- train -----------
        if self.training:
            targets = batch[1]
            preds = self.head([P3, P4, P5])
            return self.loss(preds, targets)

        return self.head([P3, P4, P5], self.args.imgsz)[0]

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = YoloLossV4To7(self, topk=1)

        return self.criterion(preds, targets)
