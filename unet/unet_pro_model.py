""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .inception_parts import *
from .unet_parts import *


class UNetPro(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetPro, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Inception_A(n_channels, 16)
        self.down1 = IDown(64, 32)
        self.down2 = IDown(128, 64)
        self.down3 = IDown(256, 128)
        factor = 2 if bilinear else 1
        self.down4 = IDown(512, 1024 // factor // 4)
        self.up1 = IUp(1024, 512 // factor // 4, bilinear)
        self.up2 = IUp(512, 256 // factor // 4, bilinear)
        self.up3 = IUp(256, 128 // factor // 4, bilinear)
        self.up4 = IUp(128, 16, bilinear)
        self.outc = IOutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
