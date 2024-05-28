import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SqueezeExcite, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        return x * out

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction_ratio=4):
        super(MBConvBlock, self).__init__()
        self.expand = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False) if expand_ratio != 1 else nn.Identity()
        self.expand_bn = nn.BatchNorm2d(in_channels * expand_ratio)
        self.depthwise_conv = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels * expand_ratio, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels * expand_ratio)
        self.se = SqueezeExcite(in_channels * expand_ratio, reduction_ratio) if expand_ratio != 1 else nn.Identity()
        self.project = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = F.relu6(self.expand_bn(self.expand(x)))
        out = F.relu6(self.depthwise_bn(self.depthwise_conv(out)))
        out = self.se(out)
        out = self.project_bn(self.project(out))
        if self.use_residual:
            out = x + out
        return out

class efficientnet_v2_s(nn.Module):
    def __init__(self, num_classes=1000):
        super(efficientnet_v2_s, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )
        self.blocks = nn.Sequential(
            MBConvBlock(24, 48, kernel_size=3, stride=1, expand_ratio=1),
            MBConvBlock(48, 96, kernel_size=3, stride=2, expand_ratio=4),
            MBConvBlock(96, 120, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(120, 240, kernel_size=3, stride=2, expand_ratio=4),
            MBConvBlock(240, 480, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(480, 672, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(672, 672, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(672, 1152, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(1152, 1152, kernel_size=3, stride=1, expand_ratio=4),
        )
        self.head = nn.Sequential(
            nn.Conv2d(1152, 1152, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1152),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1152, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out = self.head(out)
        return out


