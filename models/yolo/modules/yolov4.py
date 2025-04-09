import torch
import torch.nn as nn

from nn.backbone import Backbone
from nn.backbone import CBM
from nn.neck import SPP
from nn.head import YoloHeadV4
from models.yolo.utils.yolo_loss import YoloLossV4To7


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            CBM(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        CBM(in_filters, filters_list[0], 1),
        CBM(filters_list[0], filters_list[1], 3),
        CBM(filters_list[1], filters_list[0], 1),
        CBM(filters_list[0], filters_list[1], 3),
        CBM(filters_list[1], filters_list[0], 1),
    )
    return m


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        CBM(in_filters, filters_list[0], 1),
        CBM(filters_list[0], filters_list[1], 3),
        CBM(filters_list[1], filters_list[0], 1),
    )
    return m


class YoloV4(nn.Module):
    def __init__(self, cfg):
        super(YoloV4, self).__init__()

        scale = cfg['scale']
        scales = cfg['scales']
        width_multiple, depth_multiple = scales[scale]

        base_channels = int(width_multiple * 32)  # 64
        base_depth = max(round(depth_multiple * 3), 1)  # 3

        self.backbone = Backbone(name=f'cpsdarknetv4{scale}',
                                 layers_to_train=['stem',
                                                  'crossStagePartial1',
                                                  'crossStagePartial2',
                                                  'crossStagePartial3',
                                                  'crossStagePartial4'],
                                 return_interm_layers={'crossStagePartial2': '0',
                                                       'crossStagePartial3': '1',
                                                       'crossStagePartial4': '2', })

        self.cov1 = make_three_conv([base_channels * 16, base_channels * 32], base_channels * 32)
        self.spp = SPP([5, 9, 13])
        self.cov2 = make_three_conv([base_channels * 16, base_channels * 32], base_channels * 16 * 4)

        self.upsample1 = Upsample(base_channels * 16, base_channels * 8)
        self.conv_for_P4 = CBM(base_channels * 16, base_channels * 8, 1)
        self.make_five_conv1 = make_five_conv([base_channels * 8, base_channels * 16], base_channels * 16)

        self.upsample2 = Upsample(base_channels * 8, base_channels * 4)
        self.conv_for_P3 = CBM(base_channels * 8, base_channels * 4, 1)
        self.make_five_conv2 = make_five_conv([base_channels * 4, base_channels * 8], base_channels * 8)

        self.down_sample1 = CBM(base_channels * 4, base_channels * 8, 3, stride=2)
        self.make_five_conv3 = make_five_conv([base_channels * 8, base_channels * 16], base_channels * 16)

        self.down_sample2 = CBM(base_channels * 8, base_channels * 16, 3, stride=2)
        self.make_five_conv4 = make_five_conv([base_channels * 16, base_channels * 32], base_channels * 32)

        self.head = YoloHeadV4([base_channels * 4, base_channels * 8, base_channels * 16],
                               cfg.anchors,
                               cfg.nc)

    def forward(self, batch):
        x = batch[0]
        targets = batch[1]

        features = self.backbone(x)

        feat1, feat2, feat3 = features['0'], features['1'], features['2']

        P5 = self.cov1(feat3)
        P5 = torch.cat([P5, self.spp(P5)], 1)
        P5 = self.cov2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(feat2)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(feat1)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        P5 = self.make_five_conv4(P5)

        imgsz = self.orig_size.amax(1, True).repeat(1, 2).to(self.device)

        # ----------- train -----------
        if self.training:
            preds = self.head([P3, P4, P5], imgsz)
            return self.loss(preds, targets)

        # ----------- val -----------
        return self.head([P3, P4, P5], imgsz)[0]

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = YoloLossV4To7(self, topk=3)

        return self.criterion(preds, targets)
