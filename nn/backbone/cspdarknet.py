from typing import Tuple, Union
from functools import partial

import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

from nn.neck import SPPF, C2PSA

BN = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
CBM = partial(Conv2dNormActivation, bias=False, inplace=True, norm_layer=BN, activation_layer=nn.Mish)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut: bool = True, g=1, k: Union[Tuple[Tuple, Tuple], Tuple] = (3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBM(c1, c_, k[0], 1)
        self.cv2 = CBM(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, count=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        # c_ = c1 if first else c1 // 2
        c_ = int(c2 * e)
        self.trans_0 = CBM(c1, c_, 1)

        self.trans_1 = CBM(c1, c_, 1)

        self.make_layers = nn.Sequential()
        for _ in range(count):
            self.make_layers.append(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (1, 1)), e=1))

        self.trans_cat = CBM(c_ * 2, c2, 1)

    def forward(self, x):
        # ----------- 两分支 -----------
        out0 = self.trans_0(x)
        out1 = self.trans_1(x)

        out0 = self.make_layers(out0)

        out = torch.cat([out0, out1], 1)
        out = self.trans_cat(out)
        return out


class C2f(nn.Module):
    def __init__(self, c1, c2, count=1, shortcut=True, g=1, e=0.5):
        super(C2f, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = CBM(c1, 2 * self.c, 1, 1)
        self.cv2 = CBM((2 + count) * self.c, c2, 1)  # optional act=FReLU(c2)

        self.make_layers = nn.ModuleList()
        for _ in range(count):
            self.make_layers.append(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.make_layers)
        return self.cv2(torch.cat(y, 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.make_layers = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.make_layers = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class CSPDarknetV4(nn.Module):
    def __init__(self, base_channels=64, base_depth=3, num_classes=1000):
        super(CSPDarknetV4, self).__init__()

        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        self.stem = nn.Sequential(
            CBM(3, base_channels, 3),
            DownSampleLayer(base_channels, base_channels * 2),
            C3(base_channels * 2, base_channels * 2, base_depth * 1, e=1),
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


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = CBM(c1 * 4, c2, k)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class CSPDarknetV5(nn.Module):
    def __init__(self, base_channels=64, base_depth=3, num_classes=1000):
        super(CSPDarknetV5, self).__init__()

        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#

        CBM.keywords['activation_layer'] = nn.SiLU
        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        self.stem = CBM(3, base_channels, 6, 2)
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
            SPPF(base_channels * 16, base_channels * 16, [5], conv_layer=CBM),
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

        CBM.keywords['activation_layer'] = nn.SiLU
        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        self.stem = CBM(3, base_channels, 3, 2)

        self.crossStagePartial1 = nn.Sequential(
            DownSampleLayer(base_channels, base_channels * 2),
            C2f(base_channels * 2, base_channels * 2, base_depth),
        )

        self.crossStagePartial2 = nn.Sequential(
            DownSampleLayer(base_channels * 2, base_channels * 4),
            C2f(base_channels * 4, base_channels * 4, base_depth),
        )

        self.crossStagePartial3 = nn.Sequential(
            DownSampleLayer(base_channels * 4, base_channels * 8),
            C2f(base_channels * 8, base_channels * 8, base_depth),
        )

        self.crossStagePartial4 = nn.Sequential(
            DownSampleLayer(base_channels * 8, int(base_channels * 16 * deep_mul)),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), [5], conv_layer=CBM),
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

        CBM.keywords['activation_layer'] = nn.SiLU
        DownSampleLayer = partial(CBM, kernel_size=3, stride=2)

        self.stem = CBM(3, base_channels, 3, 2)

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
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), [5], conv_layer=CBM),
            C2PSA(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, conv_layer=CBM)
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
