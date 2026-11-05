"""CIFAR-10 ResNet-18 and three structural variants.

Baseline
    He et al. basic residual blocks with the usual CIFAR stem:
    3x3 convolution, stride 1, no max-pooling. Stage widths are
    64-128-256-512 with two blocks each (ResNet-18).

resnet18_dw
    Each 3x3 convolution *inside residual blocks* is replaced by a
    depthwise 3x3 followed by a pointwise 1x1. The stem and the
    1x1 projection shortcuts stay ordinary convolutions. Batch-norm
    and ReLU stay in the same places as the baseline, so the
    comparison isolates the convolution factorization.

resnet18_se
    A Squeeze-and-Excitation gate (Hu et al.) is inserted after the
    second batch-norm and before the residual addition. Reduction
    ratio defaults to 16. The excitation uses ReLU, as in the SE paper.

resnet18_gelu
    Every ReLU in the baseline (stem and both block activations) is
    swapped for GELU. Weight initialization is unchanged, so the
    comparison isolates the activation.

Parameter initialization follows the common CIFAR ResNet recipe:
Kaiming normal for convolutions, and the last batch-norm in each
residual block is zero-initialized so each block starts as an identity
when the shortcut is an identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

VARIANTS = ("resnet18", "resnet18_dw", "resnet18_se", "resnet18_gelu")


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class DepthwiseSeparableConv3x3(nn.Module):
    """Drop-in replacement for a bias-free 3x3 convolution.

    Depthwise 3x3 (spatial mixing per channel) then pointwise 1x1
    (channel mixing). No normalization here: the surrounding block
    still owns the single batch-norm that followed the original conv.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


def make_conv3x3(in_planes: int, out_planes: int, stride: int, separable: bool) -> nn.Module:
    if separable:
        return DepthwiseSeparableConv3x3(in_planes, out_planes, stride)
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel gate.

    Squeeze: global average pool. Excitation: two fully connected
    layers with reduction, ReLU, and a sigmoid. The gate rescales
    the residual branch before it is added to the shortcut.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("SE reduction must be >= 1")
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.act = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).reshape(b, c)
        y = self.gate(self.fc2(self.act(self.fc1(y)))).reshape(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    """Standard two-layer residual block, with optional DW conv and SE."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        separable: bool,
        use_se: bool,
        activation: str,
        se_reduction: int,
    ) -> None:
        super().__init__()
        self.conv1 = make_conv3x3(in_planes, planes, stride, separable)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = make_conv3x3(planes, planes, 1, separable)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, se_reduction) if use_se else nn.Identity()
        self.act = make_activation(activation)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.act(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """ResNet-18 skeleton adapted to 32x32 CIFAR images."""

    def __init__(
        self,
        num_classes: int = 10,
        separable: bool = False,
        use_se: bool = False,
        activation: str = "relu",
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.separable = separable
        self.use_se = use_se
        self.activation_name = activation
        self.se_reduction = se_reduction
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = make_activation(activation)
        self.layer1 = self._make_stage(64, 2, stride=1)
        self.layer2 = self._make_stage(128, 2, stride=2)
        self.layer3 = self._make_stage(256, 2, stride=2)
        self.layer4 = self._make_stage(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.reset_parameters()

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=block_stride,
                    separable=self.separable,
                    use_se=self.use_se,
                    activation=self.activation_name,
                    se_reduction=self.se_reduction,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, BasicBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def variant_kwargs(name: str, se_reduction: int = 16) -> dict:
    if name == "resnet18":
        return {"separable": False, "use_se": False, "activation": "relu", "se_reduction": se_reduction}
    if name == "resnet18_dw":
        return {"separable": True, "use_se": False, "activation": "relu", "se_reduction": se_reduction}
    if name == "resnet18_se":
        return {"separable": False, "use_se": True, "activation": "relu", "se_reduction": se_reduction}
    if name == "resnet18_gelu":
        return {"separable": False, "use_se": False, "activation": "gelu", "se_reduction": se_reduction}
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(VARIANTS)}")


def build_model(name: str, num_classes: int = 10, se_reduction: int = 16) -> ResNetCIFAR:
    if name not in VARIANTS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(VARIANTS)}")
    return ResNetCIFAR(num_classes=num_classes, **variant_kwargs(name, se_reduction))
