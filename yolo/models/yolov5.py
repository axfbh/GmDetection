import torch
import torch.nn as nn

from dataset.utils import NestedTensor

from .backbone.utils import Backbone
from .backbone.cspdarknet import CBM, C3
from .head.yolo_head import YoloHeadV5
from yolo.models.yolo_loss import YoloLossV4To7


class YoloV5(nn.Module):
    def __init__(self, cfg):
        super(YoloV5, self).__init__()

        scale = cfg['scale']
        scales = cfg['scales']
        depth_multiple, width_multiple = scales[scale]

        base_channels = int(width_multiple * 64)  # 64
        base_depth = max(round(depth_multiple * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80,80,256
        #   40,40,512
        #   20,20,1024
        # ---------------------------------------------------#
        self.backbone = Backbone(name=f'cpsdarknetv5{scale}',
                                 layers_to_train=['stem',
                                                  'crossStagePartial1',
                                                  'crossStagePartial2',
                                                  'crossStagePartial3',
                                                  'crossStagePartial4'],
                                 return_interm_layers={'crossStagePartial2': '0',
                                                       'crossStagePartial3': '1',
                                                       'crossStagePartial4': '2', })

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = CBM(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = CBM(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1 = CBM(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = CBM(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        self.head = YoloHeadV5([base_channels * 4, base_channels * 8, base_channels * 16],
                               cfg.anchors,
                               cfg.nc)

    def forward(self, batch):
        x = batch[0]
        targets = batch[1]

        features = self.backbone(x)

        feat1, feat2, feat3 = features['0'].tensors, features['1'].tensors, features['2'].tensors

        P5 = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4 = self.conv_for_feat2(P4)
        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        # ---------------------------------------------------#
        #   第三个特征层
        #   P3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   第二个特征层
        #   P4=(batch_size,75,40,40)
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   第一个特征层
        #   P5=(batch_size,75,20,20)
        # ---------------------------------------------------#

        imgsz = torch.stack([t["orig_size"].max().repeat(2) for t in targets], dim=0).to(self.device)

        # ----------- train -----------
        if self.training:
            preds = self.head([P3, P4, P5], imgsz)
            return self.loss(preds, targets)

        # ----------- val -----------
        return self.head([P3, P4, P5], imgsz)[0]

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return YoloLossV4To7(self, topk=1)

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        return self.criterion(preds, targets)
