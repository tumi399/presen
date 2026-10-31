"""Tiện ích thiết bị và seed ngẫu nhiên."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Tự động chọn thiết bị: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Trả về chuỗi mô tả thiết bị đang dùng."""
    if device.type == "cuda":
        return f"CUDA GPU: {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def set_seed(seed: int = 42) -> None:
    """Đặt seed để kết quả tái hiện được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
