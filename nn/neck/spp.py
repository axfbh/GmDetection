import torch.nn as nn
import torch
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial


class SPP(nn.Module):
    def __init__(self, ksizes=(5, 9, 13)):
        """
            SpatialPyramidPooling 空间金字塔池化, SPP 返回包含自己
        """
        super(SPP, self).__init__()
        self.make_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=(k - 1) // 2) for k in ksizes])

    def forward(self, x):
        return torch.cat([m(x) for m in self.make_layers], 1)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv3 by Glenn Jocher
    def __init__(self, c1, c2, ksizes=(5, 9, 13), conv_layer=None, activation_layer=nn.ReLU):
        super(SPPF, self).__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(ksizes) * 3 + 1), c2, 1, 1)
        self.m = SPP(ksizes)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, e=0.5, ksizes=(5, 9, 13), conv_layer=None, activation_layer=nn.ReLU):
        super(SPPCSPC, self).__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = int(2 * c2 * e)

        self.cv1 = nn.Sequential(
            Conv(c1, c_, 1),
            Conv(c_, c_, 3),
            Conv(c_, c_, 1),
        )

        self.cv2 = Conv(c1, c_, 1)

        self.spp = SPP(ksizes)

        self.cv3 = nn.Sequential(
            Conv(c_ * 4, c_, 1),
            Conv(c_, c_, 3),
        )

        self.cv4 = Conv(c_ * 2, c2, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)

        x1 = torch.cat([x1, self.spp(x1)], 1)
        x1 = self.cv3(x1)

        x = torch.cat([x1, x2], dim=1)

        return self.cv4(x)


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, conv_layer=None, activation_layer=nn.ReLU):
        super().__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, activation_layer=None)
        self.proj = Conv(dim, dim, 1, activation_layer=None)
        self.pe = Conv(dim, dim, 3, 1, groups=dim, activation_layer=None)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=False,
                       norm_layer=nn.BatchNorm2d)

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, activation_layer=None))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
