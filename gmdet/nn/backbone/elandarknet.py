import torch
import torch.nn as nn
from gmdet.nn.conv import CBS
from gmdet.nn.block import SPPCSPC


class Elan(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Elan, self).__init__()
        c_ = int(c2 * e)

        self.ids = ids
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c1, c_, 1, 1)

        self.cv3 = nn.ModuleList(
            [CBS(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)]
        )

        self.cv4 = CBS(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)

        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out


class MP1(nn.Module):
    def __init__(self, c1, c2):
        super(MP1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.cv1 = CBS(c1, c2, 1)
        self.cv2 = nn.Sequential(CBS(c1, c2, 1),
                                 CBS(c2, c2, 3, 2))

    def forward(self, x):
        x1 = self.maxpool(x)
        x1 = self.cv1(x1)

        x2 = self.cv2(x)
        return torch.cat([x1, x2], dim=1)


class ElanDarknet(nn.Module):
    def __init__(self, transition_channels=32, block_channels=32, base_depth=4, scale='l', num_classes=1000):
        super(ElanDarknet, self).__init__()

        ids = {
            'n': [-1, -3, -5, -6],
            's': [-1, -3, -5, -6],
            'm': [-1, -3, -5, -6],
            'l': [-1, -3, -5, -6],
            'x': [-1, -3, -5, -7, -8],
        }[scale]

        self.stem = nn.Sequential(
            CBS(3, transition_channels, 3),
            CBS(transition_channels, transition_channels * 2, 3, 2),
            CBS(transition_channels * 2, transition_channels * 2, 3))

        self.stage1 = nn.Sequential(
            CBS(transition_channels * 2, transition_channels * 4, 3, 2),
            Elan(transition_channels * 4, block_channels * 2, transition_channels * 8, n=base_depth, ids=ids)
        )
        self.stage2 = nn.Sequential(
            MP1(transition_channels * 8, transition_channels * 4),
            Elan(transition_channels * 8, block_channels * 4, transition_channels * 16, n=base_depth, ids=ids)
        )
        self.stage3 = nn.Sequential(
            MP1(transition_channels * 16, transition_channels * 8),
            Elan(transition_channels * 16, block_channels * 8, transition_channels * 32, n=base_depth, ids=ids)
        )
        self.stage4 = nn.Sequential(
            MP1(transition_channels * 32, transition_channels * 16),
            Elan(transition_channels * 32, block_channels * 8, transition_channels * 32, n=base_depth, ids=ids),
            SPPCSPC(transition_channels * 32, transition_channels * 16,
                    conv_layer=CBS,
                    activation_layer=nn.SiLU)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
