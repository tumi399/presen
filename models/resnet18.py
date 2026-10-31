"""ResNet-18 model definition for cats-and-dogs image classification.

Xây dựng ResNet-18 từ đầu (from scratch) không dùng pretrained weights.
Kiến trúc gốc từ bài báo:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.

Đầu vào: ảnh RGB (3 kênh) kích thước 224x224.
Đầu ra: 2 lớp — cats (0) và dogs (1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    """Khối Residual cơ bản của ResNet-18 và ResNet-34.

    Mỗi block gồm:
        - Conv 3x3 → BN → ReLU
        - Conv 3x3 → BN
        - Shortcut connection (có projection nếu kích thước thay đổi)
        - ReLU đầu ra

    expansion = 1 (không mở rộng số kênh như Bottleneck).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        # Nhánh chính
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: dùng 1x1 conv khi stride > 1 hoặc số kênh thay đổi
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # Residual connection
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 from scratch cho phân loại chó-mèo.

    Kiến trúc:
        - Stem:  Conv 7x7, stride 2 → BN → ReLU → MaxPool 3x3, stride 2
        - Layer1: 2 BasicBlock(64, 64,   stride=1)
        - Layer2: 2 BasicBlock(64, 128,  stride=2)
        - Layer3: 2 BasicBlock(128, 256, stride=2)
        - Layer4: 2 BasicBlock(256, 512, stride=2)
        - AdaptiveAvgPool → Flatten → FC(512, num_classes)

    Tổng: 18 lớp có trọng số (1 stem + 16 conv trong residual + 1 FC).
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual Layers ───────────────────────────────────────────────────
        self.layer1 = self._make_layer(in_channels=64,  out_channels=64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channels=64,  out_channels=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, n_blocks=2, stride=2)

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        # Khởi tạo trọng số (He initialization)
        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Tạo một stage gồm n_blocks BasicBlock liên tiếp."""
        blocks: list[nn.Module] = [
            BasicBlock(in_channels, out_channels, stride=stride),
        ]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        """He (Kaiming) initialization cho Conv; zeros cho BN bias."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Tensor ảnh RGB shape (N, 3, 224, 224).

        Returns:
            Logits shape (N, num_classes).
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


def resnet18(num_classes: int = 2) -> ResNet18:
    """Khởi tạo ResNet-18 với trọng số ngẫu nhiên (train from scratch)."""
    return ResNet18(num_classes=num_classes)
