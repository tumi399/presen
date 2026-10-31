"""Dataset và transforms cho bài toán phân loại chó-mèo.

Cấu trúc thư mục dataset:
    datastet/raw/
        train/cats/*.jpg
        train/dogs/*.jpg
        val/cats/*.jpg
        val/dogs/*.jpg
        test/cats/*.jpg
        test/dogs/*.jpg
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

# ─── Hằng số ──────────────────────────────────────────────────────────────────
Split = Literal["train", "val", "test"]
ModelName = Literal["lenet", "resnet18", "alexnet", "vgg16"]

CLASS_TO_IDX = {"cats": 0, "dogs": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CatsDogsDataset(Dataset):
    """Load ảnh từ ``root/<split>/{cats,dogs}``."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: Split = "train",
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        if root is None:
            root = Path(__file__).resolve().parent / "datastet" / "raw"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.class_to_idx = CLASS_TO_IDX.copy()
        self.samples: list[tuple[Path, int]] = []

        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {split_root}")

        for class_name, class_idx in self.class_to_idx.items():
            class_root = split_root / class_name
            if not class_root.is_dir():
                raise FileNotFoundError(f"Không tìm thấy thư mục lớp: {class_root}")
            images = sorted(
                p for p in class_root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise ValueError(f"Không có ảnh nào trong: {class_root}")
            self.samples.extend((p, class_idx) for p in images)

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image | Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as f:
            image = f.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(model: ModelName, split: Split, augment: bool = True) -> transforms.Compose:
    """Trả về pipeline tiền xử lý ảnh phù hợp với từng mô hình.

    - LeNet:   resize 32×32, normalize [-1, 1]
    - ResNet18: resize 256 → crop 224×224, normalize ImageNet
    """
    ops: list = []

    if split == "train" and augment:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    if model == "lenet":
        ops += [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    else:  # resnet18, alexnet, vgg16 — đều dùng 224×224, ImageNet normalization
        ops += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]

    return transforms.Compose(ops)
