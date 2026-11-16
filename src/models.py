"""
CS-FilamentNet Model Architecture
Single-Channel ResNet-34 Backbone + Multi-Directional Strip Convolutions + SE Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention module."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


class MultiDirectionalStripConv(nn.Module):
    """
    Multi-Directional Strip Convolution (MDSC) Bottleneck.
    Deploys elongated 1x15 and 15x1 strip kernels alongside dilated 3x3 atrous convolutions
    to trace thin, winding curvilinear filaments without isotropic blurring.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 4
        
        # Branch 1: Horizontal Strip (1x15)
        self.b_horiz = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Branch 2: Vertical Strip (15x1)
        self.b_vert = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(15, 1), padding=(7, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Branch 3: Dilated Atrous Conv (3x3, dilation=2)
        self.b_dilated = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Branch 4: Standard 3x3 Conv
        self.b_std = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Fusion projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Residual shortcut
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.b_horiz(x)
        v = self.b_vert(x)
        d = self.b_dilated(x)
        s = self.b_std(x)
        cat = torch.cat([h, v, d, s], dim=1)
        out = self.project(cat)
        out = self.se(out)
        return F.leaky_relu(out + res, 0.1)


class DecoderBlock(nn.Module):
    """U-Net decoder block with skip connection fusion and SE attention."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class CSFilamentNet(nn.Module):
    """
    Curvilinear Strip U-Net (CS-FilamentNet) for High-Resolution Solar Filament Segmentation.
    Backbone: ResNet-34 (adapted to 1-channel H-alpha).
    Bottleneck: Multi-Directional Strip Convolutions.
    Decoder: Skip Connections + SE Channel Attention.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)
        
        # Adapt first conv to 1-channel solar H-alpha
        old_conv = base.conv1
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            base.bn1,
            base.relu
        )
        if pretrained:
            with torch.no_grad():
                self.init_conv[0].weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  # 64 channels
        self.layer2 = base.layer2  # 128 channels
        self.layer3 = base.layer3  # 256 channels
        self.layer4 = base.layer4  # 512 channels
        
        # Bottleneck: MDSC
        self.bottleneck = MultiDirectionalStripConv(512, 512)
        
        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.init_conv(x)      # 1/2, 64
        x1 = self.maxpool(x0)       # 1/4
        x1 = self.layer1(x1)        # 1/4, 64
        x2 = self.layer2(x1)        # 1/8, 128
        x3 = self.layer3(x2)        # 1/16, 256
        x4 = self.layer4(x3)        # 1/32, 512
        
        # MDSC Bottleneck
        b = self.bottleneck(x4)     # 1/32, 512
        
        # Decoder
        d4 = self.dec4(b, x3)       # 1/16, 256
        d3 = self.dec3(d4, x2)      # 1/8, 128
        d2 = self.dec2(d3, x1)      # 1/4, 64
        d1 = self.dec1(d2, x0)      # 1/2, 32
        
        # Upsample to full input resolution
        out = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        logits = self.head(out)
        return logits
