import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NUM_CLASSES


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=64):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.u4 = DoubleConv(base * 16 + base * 8, base * 8)
        self.u3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.u2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.u1 = DoubleConv(base * 2 + base, base)

        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        cb = self.bottleneck(self.pool(c4))

        u4 = F.interpolate(cb, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.u4(torch.cat([u4, c4], dim=1))
        u3 = F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.u3(torch.cat([u3, c3], dim=1))
        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, c2], dim=1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, c1], dim=1))
        return self.out(u1)