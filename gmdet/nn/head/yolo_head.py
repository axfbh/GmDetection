import torch.nn as nn

from typing import List
from functools import partial

import torch
import math
from torchvision.ops.misc import Conv2dNormActivation

from gmdet.models.yolo.utils.misc import DFL
from gmdet.models.yolo.utils.boxes import dist2bbox
from gmdet.models.yolo.utils.ops import AnchorGenerator, make_grid


class YoloHeadV8(nn.Module):
    def __init__(self, in_channels_list: List, num_classes: int):
        super(YoloHeadV8, self).__init__()

        CBS = partial(Conv2dNormActivation, activation_layer=nn.SiLU)

        # number of classes
        self.nc = num_classes
        # number of detection layers
        self.nl = len(in_channels_list)
        # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.reg_max = 16
        # number of channel of (classes + dfl)
        self.no = num_classes + self.reg_max * 4
        # 坐标头的中间通道数
        c2 = max(16, in_channels_list[0] // 4, self.reg_max * 4)
        # 分类头的中间通道数
        c3 = max(in_channels_list[0], min(self.nc, 100))
        # 解耦头
        self.reg_head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.reg_head.append(
                nn.Sequential(
                    CBS(in_channels, c2, 3),
                    CBS(c2, c2, 3),
                    nn.Conv2d(c2, 4 * self.reg_max, 1),
                )
            )
        self.cls_head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.cls_head.append(
                nn.Sequential(
                    CBS(in_channels, c3, 3),
                    CBS(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1),
                )
            )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.anchors = AnchorGenerator((0, 0, 0), (1,))

        self.reset_parameters()

    def reset_parameters(self):
        stride = [8, 16, 32]
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.modules[-1]  # Detect() module
        for a, b, s in zip(m.reg_head, m.cls_head, stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def _inference(self, x, anchor_points, stride_tensor):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        # cxcywh
        dbox = self.decode_bboxes(self.dfl(box), anchor_points.unsqueeze(0)) * stride_tensor
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y

    def forward(self, x: List, imgsz=None):
        for i in range(self.nl):
            x[i] = torch.cat((self.reg_head[i](x[i]), self.cls_head[i](x[i])), 1)

        if self.training:  # Training path
            return x

        anchor_points, stride_tensor = (x.transpose(0, 1) for x in self.make_anchors([imgsz, imgsz], x))
        y = self._inference(x, anchor_points, stride_tensor)
        return y, x

    def make_anchors(self, image_size, preds, offset=0.5):
        # anchors : 每个尺度的[候选框左上角xy,候选框右上角xy]
        # strides : 候选框的缩放尺度
        anchors, strides = self.anchors(image_size, preds)
        # 候选框的中心点计算和尺度缩放
        for i in range(len(anchors)):
            anchors[i] = (anchors[i][..., :2] + anchors[i][..., 2:]) / 2
            anchors[i] = anchors[i] / strides[i] + offset
            strides[i] = strides[i].expand(anchors[i].shape[0], -1)
        anchor_points = torch.cat(anchors)
        strides = torch.cat(strides).flip(-1)
        return anchor_points, strides[:, 0:1]

    def decode_bboxes(self, bboxes, anchors):
        # 构建预测框 = 候选框中心点 + 预测框（左上右下）长度
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class YoloHeadV4ToV7(nn.Module):
    def __init__(self, in_channels_list: List, anchors: List, num_classes: int):
        super(YoloHeadV4ToV7, self).__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.nc = num_classes
        self.no = num_classes + 5
        self.head = nn.ModuleList()
        for in_channels in in_channels_list:
            self.head.append(nn.Conv2d(in_channels, self.na * self.no, 1, 1, 0))

        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        self.reset_parameters()

    def reset_parameters(self):
        stride = [8, 16, 32]
        for layer, s in zip(self.head, stride):
            if isinstance(layer, nn.Conv2d):
                b = layer.bias.view(self.na, -1)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.99999))
                layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: List, imgsz=None):
        for i in range(self.nl):
            x[i] = self.head[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:  # Training path
            return x

        z = self._inference(x, imgsz)
        return z, x


class YoloHeadV7(YoloHeadV4ToV7):
    def _inference(self, x, imgsz):
        z = []  # inference output
        device = self.anchors.device

        for i in range(self.nl):
            bs, _, ny, nx, _ = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            shape = 1, self.na, ny, nx, 2  # grid shape

            stride = imgsz / ny

            grid = make_grid(ny, nx, 1, 1, device).view((1, 1, ny, nx, 2)).expand(shape)
            anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape) / 640 * imgsz

            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), -1)
            xy = (xy * 3 - 1 + grid) * stride  # xy
            wh = (wh * 2) ** 2 * anchor_grid  # wh
            y = torch.cat((xy, wh, conf), 4)

            z.append(y.view(bs, self.na * nx * ny, self.no))

        return torch.cat(z, 1)


class YoloHeadV5(YoloHeadV4ToV7):
    def _inference(self, x, imgsz):
        z = []  # inference output
        device = self.anchors.device

        for i in range(self.nl):
            bs, _, ny, nx, _ = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            shape = 1, self.na, ny, nx, 2  # grid shape

            stride = imgsz / ny

            grid = make_grid(ny, nx, 1, 1, device).view((1, 1, ny, nx, 2)).expand(shape)
            anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape) / 640 * imgsz

            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), -1)
            xy = (xy * 2 - 0.5 + grid) * stride  # xy
            wh = (wh * 2) ** 2 * anchor_grid  # wh
            y = torch.cat((xy, wh, conf), 4)

            z.append(y.view(bs, self.na * nx * ny, self.no))

        return torch.cat(z, 1)


class YoloHeadV4(YoloHeadV4ToV7):
    def _inference(self, x, imgsz):
        z = []  # inference output
        device = self.anchors.device

        for i in range(self.nl):
            bs, _, ny, nx, _ = x[i].shape  # x(bs,75,20,20) to x(bs,3,20,20,25)
            shape = 1, self.na, ny, nx, 2  # grid shape

            stride = imgsz / ny

            grid = make_grid(ny, nx, 1, 1, device).view((1, 1, ny, nx, 2)).expand(shape)
            anchor_grid = self.anchors[i].view((1, self.na, 1, 1, 2)).expand(shape) / 640 * imgsz

            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), -1)
            xy = (xy + grid) * stride  # xy
            wh = (wh * 2) ** 2 * anchor_grid  # wh
            y = torch.cat((xy, wh, conf), 4)

            z.append(y.view(bs, self.na * nx * ny, self.no))

        return torch.cat(z, 1)
