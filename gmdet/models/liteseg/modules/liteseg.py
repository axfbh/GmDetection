from functools import partial

import torch
from torch import nn
from torchvision.ops.misc import ConvNormActivation

from gmdet.nn.backbone import Backbone
from gmdet.nn.conv import DepSeparableConv2d
from gmdet.nn.block import DASPP

from gmdet.utils import LOGGER


class LiteSeg(nn.Module):
    def __init__(self, cfg, nc=None):
        super(LiteSeg, self).__init__()

        self.yaml = cfg

        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value

        nc = self.yaml["nc"]

        self.backbone = Backbone(name='shufflenet_v2_x2_0',
                                 layers_to_train=['conv1',
                                                  'stage2',
                                                  'stage3',
                                                  'stage4'],
                                 return_interm_layers={'stage2': '0',
                                                       'stage4': '1'},
                                 pretrained=True)

        Conv = partial(ConvNormActivation, conv_layer=DepSeparableConv2d)

        self.aspp = DASPP(976, 96, [3, 6, 9])

        self.conv1 = Conv(96 * 5 + 976, 96, 1)

        # with some light backbone networks, apply the 1 × 1 convolution on low level features
        self.conv2 = Conv(244, 48, 1)

        self.conv3 = nn.Sequential(Conv(48 + 96, 96, 3), Conv(96, 96, 3))

        self.upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.head = nn.Conv2d(96, nc, kernel_size=1)

    def forward(self, batch):
        x = batch[0]

        feat = self.backbone(x)

        x1, x0 = feat['1'], feat['0']
        x1 = torch.cat([x1, self.aspp(x1)], dim=1)
        x1 = self.conv1(x1)
        x1 = self.upsample_4x(x1)
        x0 = self.conv2(x0)
        x2 = torch.cat([x1, x0], dim=1)
        x2 = self.conv3(x2)
        x2 = self.head(x2)
        x2 = self.upsample_8x(x2)
        return x2
