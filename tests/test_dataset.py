"""Tests cho CatsDogsDataset và get_transforms (data.py)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torch import Tensor

# Thêm thư mục gốc dự án vào path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import CLASS_TO_IDX, CatsDogsDataset, get_transforms


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        for split_i, split in enumerate(("train", "val", "test")):
            for cls_i, cls_name in enumerate(CLASS_TO_IDX):
                img_path = self.root / split / cls_name / f"{cls_name}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 32), (split_i * 50, cls_i * 100, 25)).save(img_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dataset_loads_rgb_images_with_stable_class_labels(self) -> None:
        """Load ảnh và nhãn đúng."""
        Image.new("L", (48, 32), 128).save(self.root / "train" / "cats" / "cats.png")
        dataset = CatsDogsDataset(root=self.root, split="train")

        self.assertEqual(dataset.class_to_idx, {"cats": 0, "dogs": 1})
        self.assertEqual(len(dataset), 2)
        cat_img, cat_lbl = dataset[0]
        dog_img, dog_lbl = dataset[1]
        self.assertIsInstance(cat_img, Image.Image)
        self.assertEqual(cat_img.mode, "RGB")
        self.assertEqual((cat_lbl, dog_lbl), (0, 1))

    def test_transforms_return_correct_tensor_sizes(self) -> None:
        """Transform trả về tensor đúng kích thước."""
        image = Image.new("RGB", (48, 32), (120, 80, 40))
        lenet_t  = get_transforms("lenet",   "val")(image)
        resnet_t = get_transforms("resnet18", "val")(image)

        self.assertIsInstance(lenet_t, Tensor)
        self.assertEqual(tuple(lenet_t.shape),  (3, 32, 32))
        self.assertEqual(tuple(resnet_t.shape), (3, 224, 224))

    def test_augmentation_disabled_gives_deterministic_result(self) -> None:
        """Tắt augmentation → kết quả tất định."""
        image = Image.new("RGB", (48, 32), (120, 80, 40))
        t1 = get_transforms("lenet", "train", augment=False)
        t2 = get_transforms("lenet", "val",   augment=True)
        self.assertTrue((t1(image) == t2(image)).all().item())

    def test_rejects_invalid_split(self) -> None:
        """Split không hợp lệ phải báo lỗi."""
        with self.assertRaises((ValueError, FileNotFoundError)):
            CatsDogsDataset(root=self.root, split="other")  # type: ignore

    def test_rejects_empty_class_directory(self) -> None:
        """Thư mục lớp rỗng phải báo lỗi."""
        (self.root / "val" / "dogs" / "dogs.png").unlink()
        with self.assertRaises(ValueError):
            CatsDogsDataset(root=self.root, split="val")


if __name__ == "__main__":
    unittest.main()
