"""Small U-Net and ResNet-18 U-Net baselines for document localization."""
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

class ResNet18UNet(nn.Module):
    """ImageNet-normalized encoder with full-resolution skip and frozen BN statistics.

    Encoder weights remain trainable. BatchNorm running statistics stay fixed for
    the small-batch experiment; this policy is part of the model configuration.
    """
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.full = block(3, 16)
        self.dec4 = block(512 + 256, 128)
        self.dec3 = block(128 + 128, 64)
        self.dec2 = block(64 + 64, 64)
        self.dec1 = block(64 + 64, 32)
        self.dec0 = block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    @staticmethod
    def join(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        full = self.full(x)
        stem = self.stem(x)
        a = self.layer1(self.pool(stem))
        b = self.layer2(a)
        c = self.layer3(b)
        x = self.layer4(c)
        x = self.dec4(self.join(x, c))
        x = self.dec3(self.join(x, b))
        x = self.dec2(self.join(x, a))
        x = self.dec1(self.join(x, stem))
        return self.head(self.dec0(self.join(x, full)))


def build_model(name="small", pretrained=False):
    if name == "small":
        return SmallUNet()
    if name == "resnet18":
        return ResNet18UNet(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
