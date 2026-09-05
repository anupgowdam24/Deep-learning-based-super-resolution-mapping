"""
model.py

Defines the SRUNet (Super-Resolution U-Net with PixelShuffle) architecture.
Maps 30m Sentinel-2 input (4 or 7 channels at 32x32) to 10m 5-class land-cover
logits (96x96) using a U-Net backbone and a sub-pixel convolution (PixelShuffle)
decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """[Conv2d -> BatchNorm -> ReLU] * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d followed by DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    """Upscaling with ConvTranspose2d, feature concatenation, and DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SRUNet(nn.Module):
    """
    Super-Resolution U-Net for Sentinel-2 Land-Cover Mapping.
    Input:  (B, in_channels, H_30, W_30)  e.g. (B, 4, 32, 32)
    Output: (B, num_classes, H_10, W_10) e.g. (B, 5, 96, 96)
    """
    def __init__(self, in_channels=4, num_classes=5, upscale_factor=3, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upscale_factor = upscale_factor

        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)        # 64 -> 128
        self.down2 = Down(c * 2, c * 4)    # 128 -> 256
        self.down3 = Down(c * 4, c * 8)    # 256 -> 512

        self.up1 = Up(c * 8, c * 4)        # 512 -> 256
        self.up2 = Up(c * 4, c * 2)        # 256 -> 128
        self.up3 = Up(c * 2, c)            # 128 -> 64

        # Sub-pixel convolution for 3x super-resolution
        # Output channels = num_classes * (upscale_factor ** 2) = 5 * 9 = 45
        self.out_conv = nn.Conv2d(c, num_classes * (upscale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.out_conv(x)
        logits_10m = self.pixel_shuffle(x)
        return logits_10m
