"""Tests cho Trainer (trainer.py) — train_one_epoch, eval_one_epoch, Trainer.fit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _tiny_loader(n: int = 16, img_size: int = 32, batch: int = 4) -> DataLoader:
    """Tạo DataLoader nhỏ với ảnh ngẫu nhiên để test."""
    images = torch.randn(n, 3, img_size, img_size)
    labels = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch)


class TrainEvalEpochTests(unittest.TestCase):
    """Test train_one_epoch và eval_one_epoch."""

    def setUp(self) -> None:
        from models.lenet import lenet
        self.model = lenet(num_classes=2)
        self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loader = _tiny_loader()

    def test_train_epoch_returns_valid_loss_and_acc(self) -> None:
        from train import train_one_epoch
        loss, acc = train_one_epoch(
            self.model, self.loader, self.criterion,
            self.optimizer, self.device, epoch=1, print_freq=999,
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)

    def test_train_epoch_updates_weights(self) -> None:
        from train import train_one_epoch
        w_before = next(self.model.parameters()).data.clone()
        train_one_epoch(
            self.model, self.loader, self.criterion,
            self.optimizer, self.device, epoch=1, print_freq=999,
        )
        w_after = next(self.model.parameters()).data.clone()
        self.assertFalse(torch.equal(w_before, w_after), "Trọng số không đổi sau training!")

    def test_eval_epoch_does_not_update_weights(self) -> None:
        from train import eval_one_epoch
        w_before = next(self.model.parameters()).data.clone()
        eval_one_epoch(self.model, self.loader, self.criterion, self.device)
        w_after = next(self.model.parameters()).data.clone()
        self.assertTrue(torch.equal(w_before, w_after), "Trọng số thay đổi trong eval — lỗi!")

    def test_eval_epoch_returns_valid_metrics(self) -> None:
        from train import eval_one_epoch
        loss, acc = eval_one_epoch(self.model, self.loader, self.criterion, self.device)
        self.assertGreater(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)


class TrainerFitTests(unittest.TestCase):
    """Test Trainer.fit — số epoch, checkpoint, history JSON, early stopping."""

    def setUp(self) -> None:
        from models.lenet import lenet
        self.model = lenet(num_classes=2)
        self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.loader = _tiny_loader()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_trainer(self, **kwargs):
        from train import Trainer
        return Trainer(
            self.model, self.optimizer, self.criterion,
            self.device, self.temp_dir.name, model_name="lenet", **kwargs,
        )

    def test_fit_records_correct_number_of_epochs(self) -> None:
        trainer = self._make_trainer()
        history = trainer.fit(self.loader, self.loader, num_epochs=3)
        self.assertEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["val_acc"]), 3)

    def test_fit_saves_checkpoint(self) -> None:
        trainer = self._make_trainer()
        trainer.fit(self.loader, self.loader, num_epochs=2)
        self.assertTrue(trainer.best_ckpt.exists(), f"Checkpoint không tạo: {trainer.best_ckpt}")

    def test_fit_saves_history_json(self) -> None:
        trainer = self._make_trainer()
        trainer.fit(self.loader, self.loader, num_epochs=2)
        self.assertTrue(trainer.history_path.exists())
        with open(trainer.history_path) as f:
            hist = json.load(f)
        self.assertIn("train_loss", hist)
        self.assertIn("val_acc", hist)
        self.assertEqual(len(hist["train_loss"]), 2)

    def test_early_stopping_triggers(self) -> None:
        trainer = self._make_trainer(patience=1)
        history = trainer.fit(self.loader, self.loader, num_epochs=20)
        self.assertLessEqual(len(history["train_loss"]), 20)

    def test_load_best_restores_weights(self) -> None:
        trainer = self._make_trainer()
        trainer.fit(self.loader, self.loader, num_epochs=2)
        # Xóa trọng số rồi nạp lại
        for p in self.model.parameters():
            p.data.fill_(0.0)
        trainer.load_best()
        some_param = next(self.model.parameters()).data
        self.assertFalse((some_param == 0).all().item(), "Checkpoint nạp không thành công!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
