"""PatentUNet-TAFE segmentation network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAFE(nn.Module):
    """Turbidity-aware feature enhancement module."""

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(in_channels // 4, 1)
        self.mode = "full"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def set_mode(self, mode: str):
        if mode not in {"identity", "channel", "spatial", "full"}:
            raise ValueError(f"Unsupported TAFE mode: {mode}")
        self.mode = mode

    def forward(self, x):
        if self.mode == "identity":
            return x
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        turbidity_weight = self.channel_excitation(max_out - avg_out)
        x_c = x * turbidity_weight
        if self.mode == "channel":
            return x_c

        spatial_source = x if self.mode == "spatial" else x_c
        avg_out_sp = torch.mean(spatial_source, dim=1, keepdim=True)
        max_out_sp = torch.max(spatial_source, dim=1, keepdim=True)[0]
        spatial_weight = self.sigmoid(self.spatial_conv(torch.cat([avg_out_sp, max_out_sp], dim=1)))
        return spatial_source * spatial_weight


class PatentUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(64, 128)
        self.tafe = TAFE(128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def set_tafe_mode(self, mode: str):
        self.tafe.set_mode(mode)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.tafe(self.bottleneck(self.pool2(e2)))

        d1_up = self.up1(b)
        if d1_up.shape[2:] != e2.shape[2:]:
            d1_up = F.interpolate(d1_up, size=e2.shape[2:], mode="bilinear", align_corners=True)
        d1 = self.dec1(torch.cat([d1_up, e2], dim=1))

        d2_up = self.up2(d1)
        if d2_up.shape[2:] != e1.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e1.shape[2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([d2_up, e1], dim=1))
        return self.out_conv(d2)
