"""VGG-16 cho phân loại chó-mèo (ảnh RGB 224×224).

Kiến trúc theo Simonyan & Zisserman (2014), bổ sung BatchNorm sau mỗi Conv
để ổn định training from scratch. Output = 2 classes:
    Block1: Conv 3×3 × 2 → BN → MaxPool 2×2   (64 filters)
    Block2: Conv 3×3 × 2 → BN → MaxPool 2×2   (128 filters)
    Block3: Conv 3×3 × 3 → BN → MaxPool 2×2   (256 filters)
    Block4: Conv 3×3 × 3 → BN → MaxPool 2×2   (512 filters)
    Block5: Conv 3×3 × 3 → BN → MaxPool 2×2   (512 filters)
    FC1: 25088 → 4096 → ReLU → Dropout
    FC2: 4096  → 4096 → ReLU → Dropout
    FC3: 4096  → num_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_conv_block(in_channels: int, out_channels: int, n_convs: int) -> nn.Sequential:
    """Tạo một conv block gồm n_convs lớp Conv 3×3 → BN → ReLU, sau đó MaxPool 2×2."""
    layers: list[nn.Module] = []
    for i in range(n_convs):
        layers += [
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG-16 điều chỉnh cho RGB input (3×224×224) và phân loại nhị phân."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()

        # ── Feature extractor (13 conv layers + 5 maxpool) ───────────────────
        self.features = nn.Sequential(
            _make_conv_block(3,   64,  n_convs=2),   # Block1: → 64×112×112
            _make_conv_block(64,  128, n_convs=2),   # Block2: → 128×56×56
            _make_conv_block(128, 256, n_convs=3),   # Block3: → 256×28×28
            _make_conv_block(256, 512, n_convs=3),   # Block4: → 512×14×14
            _make_conv_block(512, 512, n_convs=3),   # Block5: → 512×7×7
        )

        # ── Classifier (3 FC layers) ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                              # → 25088
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),         nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization cho Conv; zeros bias BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def vgg16(num_classes: int = 2) -> VGG16:
    """Khởi tạo VGG-16 với trọng số ngẫu nhiên (train from scratch)."""
    return VGG16(num_classes=num_classes)
