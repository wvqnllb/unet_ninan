import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积层
class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BN_Conv2d,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding = padding, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))


class Inception_A(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, out_channels, 1, 1, 0)
        )
        self.branch2 = BN_Conv2d(in_channels, out_channels, 1, 1, 0)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, in_channels, 1, 1, 0),
            BN_Conv2d(in_channels, out_channels, 3, 1, 1),
            BN_Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, in_channels, 1, 1, 0),
            BN_Conv2d(in_channels, out_channels, 5, 1, 2)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class IConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(IConv, self).__init__()
        self.inception = Inception_A(in_channels, out_channels)
        self.conv = BN_Conv2d(out_channels * 4, out_channels * 4, 3, 1, 1)

    def forward(self, x):
        x1 = self.inception(x)
        x2 = F.relu(x1)
        x3 = self.conv(x2)
        return x3

class IDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            IConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class IUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.se = SELayer(in_channels // 2)
            self.conv = IConv(in_channels, out_channels)
            self.c = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = IConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3 = self.c(x2)
        x3 = self.se(x3)
        x3 = x3 + x2
        x4 = F.relu(x3)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x4, x1], dim=1)
        return self.conv(x)


class IOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)