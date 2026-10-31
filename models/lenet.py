"""LeNet-5 cho phân loại chó-mèo (ảnh RGB 32×32).

Kiến trúc:
    Conv1: 3→6, kernel 5×5  → ReLU → MaxPool 2×2
    Conv2: 6→16, kernel 5×5 → ReLU → MaxPool 2×2
    FC1: 400 → 120 → ReLU
    FC2: 120 → 84  → ReLU
    FC3: 84  → num_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet-5 điều chỉnh cho RGB input (3×32×32) và phân loại nhị phân."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),   # → (6, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # → (6, 14, 14)

            nn.Conv2d(6, 16, kernel_size=5),  # → (16, 10, 10)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),               # → (16, 5, 5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), 
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),          
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def lenet(num_classes: int = 2) -> LeNet:
    """Khởi tạo LeNet với trọng số ngẫu nhiên (train from scratch)."""
    return LeNet(num_classes=num_classes)
