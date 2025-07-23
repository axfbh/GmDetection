import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from gmdet.nn.backbone import Backbone
from gmdet.nn.conv import CBM
from gmdet.nn.block import ASPP

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
                                                  'stage4',
                                                  'conv5'],
                                 return_interm_layers={'maxpool': '0',
                                                       'conv5': '1'},
                                 pretrained=True,
                                 norm_layer=FrozenBatchNorm2d)

        self.aspp = ASPP(2048, 96, [3, 6, 9])

        self.conv1 = CBM(96 * 5 + 2048, 96, 1)

        self.conv3 = nn.Sequential(CBM(24 + 96, 96, 3), CBM(96, 96, 3))

        self.upsample_8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.head = nn.Conv2d(96, nc, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)

        x1, x0 = feat['1'], feat['0']
        x1 = torch.cat([x1, self.aspp(x1)], dim=1)
        x1 = self.conv1(x1)
        x1 = self.upsample_8x(x1)
        x2 = torch.cat([x1, x0], dim=1)
        x2 = self.conv3(x2)
        x2 = self.head(x2)
        x2 = self.upsample_4x(x2)
        return x2
