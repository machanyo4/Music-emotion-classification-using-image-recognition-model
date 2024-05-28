import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from typing import List

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

class FusedMBConv(nn.Module):
    def __init__(
        self,
        stride: int,
        input_channels: int,
        out_channels: int,
        expand_ratio: int,
        stochastic_depth_prob: float
    ) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = input_channels * expand_ratio
        if expanded_channels != input_channels:
            # fused expand
            layers.extend([
                nn.Conv2d(input_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expanded_channels),
                activation_layer()
            ])

            # project
            layers.extend([
                nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
        else:
            layers.extend([
                nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation_layer()
            ])

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

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
            FusedMBConv(1, 24, 24, 2, 0.2),
            FusedMBConv(2, 24, 48, 4, 0.2),
            FusedMBConv(2, 48, 64, 4, 0.2),
            MBConvBlock(64, 128, kernel_size=3, stride=2, expand_ratio=4),
            MBConvBlock(128, 160, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(160, 256, kernel_size=3, stride=2, expand_ratio=4),
        )
        self.head = nn.Sequential(
            MBConvBlock(256, 512, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(512, 512, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(512, 1152, kernel_size=3, stride=1, expand_ratio=4),
            MBConvBlock(1152, 1152, kernel_size=3, stride=1, expand_ratio=4),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1152, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out = self.head(out)
        out = self.classifier(out)
        return out
