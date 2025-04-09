from typing import List

import torch.nn as nn
import torch
import torch.nn as nn

from nn.backbone import ElanDarknet, CBS, MP1, Elan
from nn.neck import SPPCSPC
from nn.head import YoloHeadV7
from nn.backbone.ops import Backbone

from models.yolo.utils.yolo_loss import YoloLossV4To7


class YoloV7(nn.Module):
    def __init__(self, cfg):
        super(YoloV7, self).__init__()

        scale = cfg['scale']
        scales = cfg['scales']
        multiple = scales[scale]

        base_depth = max(round(multiple[0] * 4), 4)
        transition_channels = int(multiple[1] * 32)  # 64
        block_channels = int(multiple[2] * 32)  # 64
        e = int(multiple[3] * 2)  # 64

        # base_depth = {'l': 4, 'x': 6}[scales]
        # transition_channels = {'l': 32, 'x': 40}[scales]
        # block_channels = {'l': 32, 'x': 64}[scales]
        # e = {'l': 2, 'x': 1}[scales]
        ids = {'n': [-1, -2, -3, -4, -5, -6],
               's': [-1, -2, -3, -4, -5, -6],
               'm': [-1, -2, -3, -4, -5, -6],
               'l': [-1, -2, -3, -4, -5, -6],
               'x': [-1, -3, -5, -7, -8]}[scale]

        self.backbone = Backbone(name='ElanDarknet',
                                 layers_to_train=['stem',
                                                  'stage1',
                                                  'stage2',
                                                  'stage3',
                                                  'stage4'],
                                 return_interm_layers={'stage2': '0',
                                                       'stage3': '1',
                                                       'stage4': '2'},
                                 transition_channels=transition_channels,
                                 block_channels=block_channels,
                                 base_depth=base_depth,
                                 scale=scale)

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16,
                               conv_layer=CBS,
                               activation_layer=nn.SiLU)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_P5 = CBS(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = CBS(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Elan(transition_channels * 16, block_channels * 4, transition_channels * 8,
                                        e=e, n=base_depth, ids=ids)

        self.conv_for_P4 = CBS(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = CBS(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Elan(transition_channels * 8, block_channels * 2, transition_channels * 4,
                                        e=e, n=base_depth, ids=ids)

        self.down_sample1 = MP1(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Elan(transition_channels * 16, block_channels * 4, transition_channels * 8,
                                          e=e, n=base_depth, ids=ids)

        self.down_sample2 = MP1(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Elan(transition_channels * 32, block_channels * 8, transition_channels * 16,
                                          e=e, n=base_depth, ids=ids)

        self.rep_conv_1 = CBS(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = CBS(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = CBS(transition_channels * 16, transition_channels * 32, 3, 1)

        self.head = YoloHeadV7([transition_channels * 8, transition_channels * 16, transition_channels * 32],
                               cfg.anchors,
                               cfg.nc)

    def forward(self, batch):
        x = batch[0]

        features = self.backbone(x)

        feat1, feat2, feat3 = features['0'], features['1'], features['2']

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        imgsz = self.orig_size.amax(1, True).repeat(1, 2).to(self.device)

        # ----------- train -----------
        if self.training:
            targets = batch[1]
            preds = self.head([P3, P4, P5], imgsz)
            return self.loss(preds, targets)

        # ----------- val -----------
        return self.head([P3, P4, P5], imgsz)[0]

    def loss(self, preds, targets):
        if getattr(self, "criterion", None) is None:
            self.criterion = YoloLossV4To7(self, topk=5)

        return self.criterion(preds, targets)
