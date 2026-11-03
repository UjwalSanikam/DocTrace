"""Small U-Net for pipeline checks; randomly initialized, no downloads."""
import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = block(3, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.bridge = block(base * 4, base * 8)
        self.dec3 = block(base * 12, base * 4)
        self.dec2 = block(base * 6, base * 2)
        self.dec1 = block(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        x = self.bridge(F.max_pool2d(c, 2))
        x = self.dec3(torch.cat([F.interpolate(x, size=c.shape[-2:], mode="bilinear", align_corners=False), c], 1))
        x = self.dec2(torch.cat([F.interpolate(x, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        x = self.dec1(torch.cat([F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        return self.head(x)
