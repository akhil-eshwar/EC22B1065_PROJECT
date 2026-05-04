# ==========================================================
# unetpp_model.py
# Optimized U-Net++ for 4GB GPU
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
# Conv Block
# ----------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ----------------------------------------------------------
# U-Net++
# ----------------------------------------------------------
class UNetPP(nn.Module):

    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()

        # Reduced channels for low VRAM GPU
        nb = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv0_0 = ConvBlock(in_channels, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4])

        self.conv0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.conv1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.conv2_1 = ConvBlock(nb[2] + nb[3], nb[2])
        self.conv3_1 = ConvBlock(nb[3] + nb[4], nb[3])

        self.conv0_2 = ConvBlock(nb[0]*2 + nb[1], nb[0])
        self.conv1_2 = ConvBlock(nb[1]*2 + nb[2], nb[1])
        self.conv2_2 = ConvBlock(nb[2]*2 + nb[3], nb[2])

        self.conv0_3 = ConvBlock(nb[0]*3 + nb[1], nb[0])
        self.conv1_3 = ConvBlock(nb[1]*3 + nb[2], nb[1])

        self.conv0_4 = ConvBlock(nb[0]*4 + nb[1], nb[0])

        self.final = nn.Conv2d(nb[0], n_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat(
            [x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return self.final(x0_4)