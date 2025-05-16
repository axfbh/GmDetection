import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial

from nn.conv import CBS


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, c_, k[0], 1)
        self.cv2 = CBS(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C0(nn.Module):
    def __init__(self, c, n=1, shortcut=True, g=1):
        super().__init__()
        self.m = nn.Sequential(*(Bottleneck(c, c, shortcut, g, k=((3, 3), (1, 1))) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.m(x)


class C1(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1):
        super().__init__()
        self.cv1 = CBS(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c2, c2, shortcut, g, k=((3, 3), (1, 1)), e=2.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.m(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c1, c_, 1, 1)
        self.cv3 = CBS(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, 2 * self.c, 1, 1)
        self.cv2 = CBS((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, conv_layer=None, activation_layer=nn.ReLU):
        super().__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=True,
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
                       inplace=True,
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
                       inplace=True,
                       norm_layer=nn.BatchNorm2d)

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, activation_layer=None))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class SPP(nn.Module):
    def __init__(self, k=(5, 9, 13), add=False):
        """
            SpatialPyramidPooling 空间金字塔池化, SPP 返回包含自己
        """
        super(SPP, self).__init__()
        self.add = add
        self.make_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=(k - 1) // 2) for k in k])

    def forward(self, x):
        return torch.cat([x] + [m(x) for m in self.make_layers], 1) if self.add else torch.cat(
            [m(x) for m in self.make_layers], 1)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv3 by Glenn Jocher
    def __init__(self, c1, c2, k=(5, 9, 13), conv_layer=None, activation_layer=nn.ReLU):
        super(SPPF, self).__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=True,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) * 3 + 1), c2, 1, 1)
        self.m = SPP(k)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13), conv_layer=None, activation_layer=nn.ReLU):
        super(SPPCSPC, self).__init__()

        Conv = partial(Conv2dNormActivation,
                       bias=False,
                       inplace=True,
                       norm_layer=nn.BatchNorm2d,
                       activation_layer=activation_layer) if conv_layer is None else conv_layer

        c_ = int(2 * c2 * e)

        self.cv1 = nn.Sequential(
            Conv(c1, c_, 1),
            Conv(c_, c_, 3),
            Conv(c_, c_, 1),
        )

        self.cv2 = Conv(c1, c_, 1)

        self.spp = SPP(k)

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
