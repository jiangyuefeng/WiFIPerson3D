import torch
from .BasicModel import BasicModel
from torch import nn
from torch.nn import functional
from torch.nn import init
from .UNet_3Plus import UNet_3Plus

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


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups
    )


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels)
        )


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1
    )


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        # print('downConv', x.shape)
        x = functional.relu(self.conv2(x))
        # print('downConv', x.shape)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = functional.relu(self.conv1(x))
        # print('upConv', x.shape)
        x = functional.relu(self.conv2(x))
        # print('upConv', x.shape)
        return x




class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10

        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):

        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()  # shape[b,1,38,38]
        x = x / norm   # shape[b,512,38,38]

        out = self.weight[None, ..., None, None] * x
        return out


class WiFiModel3(BasicModel):
    def __init__(self, num_classes=150, in_channels=150):
        super(WiFiModel3, self).__init__()
        self.model_name = 'WiFiModel3'

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.upsample = nn.Upsample(mode='bilinear', scale_factor=32, align_corners=False)
        self.residual = ResidualBlock(self.in_channels, in_channels, stride=1, shortcut=None)
        self.UNet1 = UNet_3Plus(self.in_channels, self.num_classes, feature_scale=4)
        # self.downsample = nn.AdaptiveMaxPool2d((46, 82))
        self.fullconv_1 = nn.Sequential(
            nn.AdaptiveMaxPool2d((46, 82)),

            nn.Conv2d(150, 52, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(52),
            nn.ReLU(inplace=True),

            nn.Conv2d(52, 26, kernel_size=1, stride=1, padding=0, bias=False),
            # L2Norm(26, 10),
            nn.BatchNorm2d(26),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fullconv_2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((46, 82)),

            nn.Conv2d(150, 104, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(104),
            nn.ReLU(inplace=True),

            nn.Conv2d(104, 52, kernel_size=1, stride=1, padding=0, bias=False),
            # L2Norm(52, 10),
            nn.BatchNorm2d(52),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = self.upsample(x)
        x = functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        x = self.residual(x)
        jhm = self.UNet1(x)
        # print(jhm.shape)
        # jhm = self.downsample(jhm)
        jhm = self.fullconv_1(jhm)


        return sm_jhm
