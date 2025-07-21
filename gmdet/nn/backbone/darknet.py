from functools import partial

import torch
from torch import nn

from gmdet.nn.conv import CBS
from gmdet.nn.block import C0


class Darknet(nn.Module):
    def __init__(self, base_channels=64, base_depth=2, num_classes=1000):
        super(Darknet, self).__init__()

        DownSampleLayer = partial(CBS, kernel_size=3, stride=2)

        # ----- DarkNet53 ------------
        self.stem = nn.Sequential(
            CBS(3, base_channels, 3, 1, 1),
            DownSampleLayer(base_channels, base_channels * 2),
            C0(base_channels * 2, 1),
        )

        self.feature1 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C0(base_channels * 4, base_depth),
        )

        self.feature2 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C0(base_channels * 8, base_depth * 4)
        )

        self.feature3 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            C0(base_channels * 16, base_depth * 4)
        )

        self.feature4 = nn.Sequential(
            DownSampleLayer(base_channels * 16, base_channels * 32),
            C0(base_channels * 32, base_depth * 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 32, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
