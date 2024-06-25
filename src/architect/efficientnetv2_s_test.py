import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, norm_layer=None, activation_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channels),
            activation_layer(inplace=True)
        ]
        super(Conv2dNormActivation, self).__init__(*layers)

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = input_channels // squeeze_factor
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio=4, se_ratio=0.25, stochastic_depth_prob=0.0):
        super(FusedMBConv, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.stochastic_depth_prob = stochastic_depth_prob

        expanded_channels = in_channels * expand_ratio
        self.block = nn.Sequential(
            Conv2dNormActivation(in_channels, expanded_channels, kernel_size, stride, padding),
            Conv2dNormActivation(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, activation_layer=None)
        )
        self.se = SqueezeExcitation(expanded_channels, int(1 / se_ratio))

    def forward(self, x):
        result = self.block(x)
        result = self.se(result)
        if self.use_residual:
            if self.stochastic_depth_prob > 0.0 and self.training:
                result = self.stochastic_depth(result, self.stochastic_depth_prob)
            result = result + x
        return result

    @staticmethod
    def stochastic_depth(x, prob):
        if not x.is_cuda:
            return x
        keep_prob = 1 - prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class EfficientNetV2_s(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetV2_s, self).__init__()
        self.features = nn.Sequential(
            Conv2dNormActivation(1, 24, kernel_size=3, stride=2, padding=1),
            self._make_layer(24, 24, 2, 3, 1, 1, stochastic_depth_prob=0.005),
            self._make_layer(24, 48, 4, 3, 2, 1, stochastic_depth_prob=0.01),
            self._make_layer(48, 64, 4, 3, 2, 1, stochastic_depth_prob=0.015),
            self._make_layer(64, 128, 6, 3, 2, 1, stochastic_depth_prob=0.02),
            self._make_layer(128, 256, 9, 3, 2, 1, stochastic_depth_prob=0.025),
            self._make_layer(256, 512, 15, 3, 2, 1, stochastic_depth_prob=0.03),
            Conv2dNormActivation(512, 1280, kernel_size=1, stride=1, padding=0)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, kernel_size, stride, padding, stochastic_depth_prob):
        layers = []
        for i in range(num_blocks):
            layers.append(
                FusedMBConv(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    padding=padding,
                    stochastic_depth_prob=stochastic_depth_prob
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x