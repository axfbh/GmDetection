import torch
import torch.nn as nn

from nn.conv import CBS
from nn.block import C3
from nn.backbone import Backbone
from nn.head import YoloHeadV5
from models.yolo.utils.yolo_loss import YoloLossV4To7


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
        self.backbone = Backbone(name='CSPDarknetV5',
                                 layers_to_train=['stem',
                                                  'crossStagePartial1',
                                                  'crossStagePartial2',
                                                  'crossStagePartial3',
                                                  'crossStagePartial4'],
                                 return_interm_layers={'crossStagePartial2': '0',
                                                       'crossStagePartial3': '1',
                                                       'crossStagePartial4': '2'},
                                 base_channels=base_channels,
                                 base_depth=base_depth)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = CBS(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16,
                                      base_channels * 8,
                                      base_depth,
                                      shortcut=False)

        self.conv_for_feat2 = CBS(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8,
                                      base_channels * 4,
                                      base_depth,
                                      shortcut=False)

        self.down_sample1 = CBS(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(base_channels * 8,
                                        base_channels * 8,
                                        base_depth,
                                        shortcut=False)

        self.down_sample2 = CBS(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(base_channels * 16,
                                        base_channels * 16,
                                        base_depth,
                                        shortcut=False)

        self.head = YoloHeadV5([base_channels * 4, base_channels * 8, base_channels * 16],
                               cfg.anchors,
                               cfg.nc)

    def forward(self, batch):
        x = batch[0]

        features = self.backbone(x)

        feat1, feat2, feat3 = features['0'], features['1'], features['2']

        P5 = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        # ---------------------------------------------------#
        #   第三个特征层
        #   P3=(batch_size,75,80,80)
        # ---------------------------------------------------#
        P4 = self.conv_for_feat2(P4)
        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        # ---------------------------------------------------#
        #   第二个特征层
        #   P4=(batch_size,75,40,40)
        # ---------------------------------------------------#
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        # ---------------------------------------------------#
        #   第一个特征层
        #   P5=(batch_size,75,20,20)
        # ---------------------------------------------------#
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        # ----------- train -----------
        if self.training:
            targets = batch[2]
            preds = self.head([P3, P4, P5])
            return self.loss(preds, targets)

        orig_size = batch[1]
        # ----------- val -----------
        scales = orig_size.amax(1, True) / self.args.imgsz

        preds = self.head([P3, P4, P5], self.args.imgsz)[0]

        # ------------- 满足 coco eval -------------
        preds[:, :, :4] *= scales[:, None]
        return preds

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = YoloLossV4To7(self, topk=3)

        return self.criterion(preds, targets)
