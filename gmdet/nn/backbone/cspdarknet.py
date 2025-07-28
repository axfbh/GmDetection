from functools import partial

import torch
import torch.nn as nn

from gmdet.nn.conv import CBS
from gmdet.nn.block import C2PSA, C3, C3k2, C2f, C1, SPPF, SPP


class CSPDarknetV4(nn.Module):
    def __init__(self, base_channels=64, base_depth=3, num_classes=1000):
        super(CSPDarknetV4, self).__init__()

        DownSampleLayer = partial(CBS, kernel_size=3, stride=2)

        self.stem = nn.Sequential(
            CBS(3, base_channels, 3),
            DownSampleLayer(base_channels, base_channels * 2),
            C3(base_channels * 2, base_channels * 2, base_depth, e=1),
        )

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C3(base_channels * 8, base_channels * 8, base_depth * 8),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth * 8),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 16, base_channels * 32),
            C3(base_channels * 32, base_channels * 32, base_depth * 4),
            C1(base_channels * 32, base_channels * 16, 1, shortcut=False),
            SPP([5, 9, 13], add=True),
            C1(base_channels * 16 * 4, base_channels * 16, 1, shortcut=False),
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
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CSPDarknetV5(nn.Module):
    def __init__(self, base_channels=64, base_depth=3, num_classes=1000):
        super(CSPDarknetV5, self).__init__()

        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        DownSampleLayer = partial(CBS, kernel_size=3, stride=2)

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = CBS(3, base_channels, 6, 2)
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16, [5], activation_layer=nn.SiLU),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 16, num_classes)

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
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CSPDarknetV8(nn.Module):
    def __init__(self, base_channels: int = 64, base_depth: int = 3, deep_mul=1.0, num_classes=1000):
        super(CSPDarknetV8, self).__init__()

        DownSampleLayer = partial(CBS, kernel_size=3, stride=2)

        self.stem = CBS(3, base_channels, 3, 2)

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, shortcut=True),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C2f(base_channels * 4, base_channels * 4, base_depth, shortcut=True),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C2f(base_channels * 8, base_channels * 8, base_depth, shortcut=True),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, int(base_channels * 16 * deep_mul)),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, shortcut=True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), [5], activation_layer=nn.SiLU),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 16, num_classes)

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
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CSPDarknetV11(nn.Module):
    def __init__(self, base_channels: int = 64, base_depth: int = 3, deep_mul=1.0, num_classes=1000):
        super(CSPDarknetV11, self).__init__()

        DownSampleLayer = partial(CBS, kernel_size=3, stride=2)

        self.stem = CBS(3, base_channels, 3, 2)

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            C3k2(base_channels * 2, base_channels * 4, base_depth, c3k=False, e=0.25),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 4),
            C3k2(base_channels * 4, base_channels * 8, base_depth, c3k=False, e=0.25),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 8, base_channels * 8),
            C3k2(base_channels * 8, base_channels * 8, base_depth, c3k=True),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, int(base_channels * 16 * deep_mul)),
            C3k2(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, c3k=True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), [5], activation_layer=nn.SiLU),
            C2PSA(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth,
                  activation_layer=nn.SiLU)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 16, num_classes)

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
        x = self.crossStagePartial1(x)
        x = self.crossStagePartial2(x)
        x = self.crossStagePartial3(x)
        x = self.crossStagePartial4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
