"""AlexNet cho phân loại chó-mèo (ảnh RGB 224×224).

Kiến trúc theo Krizhevsky et al. (2012), bổ sung BatchNorm sau mỗi Conv
để ổn định training from scratch. Output = 2 classes:
    Conv1: 3→96,  kernel 11×11, stride 4  → BN → ReLU → MaxPool 3×3 stride 2
    Conv2: 96→256, kernel 5×5,  padding 2 → BN → ReLU → MaxPool 3×3 stride 2
    Conv3: 256→384, kernel 3×3, padding 1 → BN → ReLU
    Conv4: 384→384, kernel 3×3, padding 1 → BN → ReLU
    Conv5: 384→256, kernel 3×3, padding 1 → BN → ReLU → MaxPool 3×3 stride 2
    FC1: 9216 → 4096 → ReLU → Dropout
    FC2: 4096 → 4096 → ReLU → Dropout
    FC3: 4096 → num_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet điều chỉnh cho RGB input (3×224×224) và phân loại nhị phân."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()

        # ── Feature extractor (5 conv layers + BatchNorm) ────────────────────
        self.features = nn.Sequential(
            # Conv1: 3×224×224 → 96×55×55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # → 96×27×27

            # Conv2: 96×27×27 → 256×27×27
            nn.Conv2d(96, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # → 256×13×13

            # Conv3: 256×13×13 → 384×13×13
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Conv4: 384×13×13 → 384×13×13
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Conv5: 384×13×13 → 256×13×13
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # → 256×6×6
        )

        # ── Classifier (3 FC layers) ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                                    # → 9216
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),         nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization cho Conv và BN."""
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


def alexnet(num_classes: int = 2) -> AlexNet:
    """Khởi tạo AlexNet với trọng số ngẫu nhiên (train from scratch)."""
    return AlexNet(num_classes=num_classes)
