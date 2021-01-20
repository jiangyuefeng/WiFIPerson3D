import torch
from .BasicModel import BasicModel
from torch import nn
from torch.nn import functional
from.unet import unet


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return functional.relu(out)


class csipose(BasicModel):
    def __init__(self, num_classes=300, in_channels=150):
        super(csipose, self).__init__()
        self.model_name = 'csipose'

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.upsample = nn.Upsample(mode='bilinear', scale_factor=32, align_corners=False)
        self.residual = ResidualBlock(self.in_channels, in_channels, stride=1, shortcut=None)
        self.UNet = unet(feature_scale=1, n_classes=self.num_classes, in_channels=self.in_channels)
        self.downsample = nn.AdaptiveAvgPool2d((46, 82))
        self.fullconv_1 = nn.Sequential(
            nn.Conv2d(self.num_classes, 52, 3, 1, 1),
            nn.BatchNorm2d(52),
            nn.ReLU(inplace=True),
            nn.Conv2d(52, 26, 1, 1),
            nn.BatchNorm2d(26),
            nn.Sigmoid()

        )
        self.fullconv_2 = nn.Sequential(
            nn.Conv2d(self.num_classes, 104, 3, 1, 1),
            nn.BatchNorm2d(104),
            nn.ReLU(inplace=True),
            nn.Conv2d(104, 52, 1, 1),
            nn.BatchNorm2d(52)
        )

    def forward(self, x):
        # x = self.upsample(x)
        x = functional.interpolate(x, scale_factor=64, mode='bilinear', align_corners=False)
        x = self.residual(x)

        sm_jhm = self.UNet(x)
        sm_jhm = self.downsample(sm_jhm)
        sm_jhm = self.fullconv_1(sm_jhm)

        pafs = self.UNet(x)
        pafs = self.downsample(pafs)
        pafs = self.fullconv_2(pafs)

        return (sm_jhm, pafs)
