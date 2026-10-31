"""Tests cho model definitions (models/lenet.py và models/resnet18.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LeNetModelTests(unittest.TestCase):
    def setUp(self) -> None:
        from models.lenet import lenet
        self.model = lenet(num_classes=2)
        self.model.eval()

    def test_output_shape(self) -> None:
        """Output shape phải là (N, 2) với input (N, 3, 32, 32)."""
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (4, 2))

    def test_single_image_forward(self) -> None:
        """Predict một ảnh đơn (batch=1) không lỗi."""
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (1, 2))

    def test_parameter_count_under_100k(self) -> None:
        """LeNet phải có dưới 100,000 tham số."""
        n = sum(p.numel() for p in self.model.parameters())
        self.assertLess(n, 100_000, f"LeNet có {n:,} tham số — quá nhiều!")

    def test_no_nan_or_inf_in_output(self) -> None:
        """Output không chứa NaN hay Inf."""
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            out = self.model(x)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

    def test_wrong_input_size_raises(self) -> None:
        """Input 224×224 phải báo lỗi với LeNet."""
        with self.assertRaises(Exception):
            self.model(torch.randn(2, 3, 224, 224))


class ResNet18ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        from models.resnet18 import resnet18
        self.model = resnet18(num_classes=2)
        self.model.eval()

    def test_output_shape(self) -> None:
        """Output shape phải là (N, 2) với input (N, 3, 224, 224)."""
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (4, 2))

    def test_parameter_count_exceeds_lenet(self) -> None:
        """ResNet18 phải nhiều tham số hơn LeNet đáng kể."""
        from models.lenet import lenet
        lenet_n   = sum(p.numel() for p in lenet(num_classes=2).parameters())
        resnet_n  = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(resnet_n, lenet_n * 10)

    def test_parameter_count_around_11m(self) -> None:
        """ResNet18 chuẩn có khoảng 11.2M tham số."""
        n = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n, 9_000_000)
        self.assertLess(n, 13_000_000)

    def test_no_nan_or_inf_in_output(self) -> None:
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

    def test_residual_blocks_count(self) -> None:
        """ResNet18 phải có đúng 8 BasicBlock."""
        from models.resnet18 import BasicBlock
        n = sum(1 for m in self.model.modules() if isinstance(m, BasicBlock))
        self.assertEqual(n, 8)

    def test_weights_not_all_zero(self) -> None:
        """Trọng số phải được khởi tạo (không toàn 0)."""
        for name, param in self.model.named_parameters():
            if "weight" in name and param.numel() > 1:
                self.assertFalse((param == 0).all().item())
                break


class ModelPackageTests(unittest.TestCase):
    def test_models_package_exports(self) -> None:
        """models/lenet.py và models/resnet18.py phải export đúng class."""
        from models.lenet   import LeNet, lenet
        from models.resnet18 import ResNet18, resnet18
        self.assertIsInstance(lenet(num_classes=2),    LeNet)
        self.assertIsInstance(resnet18(num_classes=2), ResNet18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
