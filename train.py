# -*- coding: utf-8 -*-
"""Train mô hình CNN trên dataset chó-mèo.

Hỗ trợ: LeNet, ResNet18, AlexNet, VGG-16.

Chứa toàn bộ training logic (Trainer class, epoch functions) và CLI script.

Import trong notebook/code khác:
    from train import Trainer, train_one_epoch, eval_one_epoch

Chạy từ terminal:
    python train.py --model lenet    --epochs 30 --batch-size 64
    python train.py --model resnet18 --epochs 30 --batch-size 32
    python train.py --model alexnet  --epochs 30 --batch-size 32
    python train.py --model vgg16    --epochs 30 --batch-size 16
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Vòng lặp một epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 100,
) -> tuple[float, float]:
    """Train một epoch, trả về (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        n = images.size(0)
        total_loss += loss.item() * n
        correct += (outputs.argmax(1) == labels).sum().item()
        total += n

        if (batch_idx + 1) % print_freq == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(loader)}"
                  f" | Loss: {total_loss/total:.4f} | Acc: {correct/total*100:.2f}%")

    return total_loss / total, correct / total * 100


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Đánh giá một epoch (không cập nhật gradient), trả về (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        n = images.size(0)
        total_loss += loss.item() * n
        correct += (outputs.argmax(1) == labels).sum().item()
        total += n

    return total_loss / total, correct / total * 100


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """Điều phối toàn bộ quá trình train: loop epoch, checkpoint, early stopping.

    Args:
        model:      Mô hình CNN (LeNet hoặc ResNet18).
        optimizer:  Optimizer đã khởi tạo.
        criterion:  Hàm mất mát (thường là CrossEntropyLoss).
        device:     Thiết bị tính toán.
        output_dir: Thư mục lưu checkpoint và history JSON.
        model_name: Tên mô hình, dùng để đặt tên file.
        scheduler:  LR scheduler (tùy chọn).
        patience:     Early stopping — số epoch không cải thiện. None = tắt.
        save_best_by: Tiêu chí lưu checkpoint tốt nhất ('val_acc' hoặc 'val_loss').
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        output_dir: str | Path,
        model_name: str = "model",
        scheduler: _LRScheduler | None = None,
        patience: int | None = None,
        save_best_by: str = "val_acc",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.scheduler = scheduler
        self.patience = patience
        self.save_best_by = save_best_by

        if self.save_best_by not in ("val_acc", "val_loss"):
            raise ValueError("save_best_by phải là 'val_acc' hoặc 'val_loss'")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self._best_metric = (
            float("-inf") if self.save_best_by == "val_acc"
            else float("inf")
        )
        self._no_improve = 0

    @property
    def best_ckpt(self) -> Path:
        return self.output_dir / f"{self.model_name}_best.pth"

    @property
    def history_path(self) -> Path:
        return self.output_dir / f"{self.model_name}_history.json"

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
    ) -> dict:
        """Chạy toàn bộ quá trình training, trả về history dict."""
        print(f"\n{'='*55}")
        print(f"  Training {self.model_name.upper()} tren {self.device}")
        print(f"  Epochs: {num_epochs}  |  Output: {self.output_dir}")
        print(f"{'='*55}\n")

        self.model.to(self.device)
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                self.model, train_loader, self.criterion,
                self.optimizer, self.device, epoch, print_freq=200,
            )
            val_loss, val_acc = eval_one_epoch(
                self.model, val_loader, self.criterion, self.device,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:3d}/{num_epochs}"
                  f" | Train  Loss {train_loss:.4f}  Acc {train_acc:.2f}%"
                  f" | Val  Loss {val_loss:.4f}  Acc {val_acc:.2f}%")

            current_metric = (
                val_acc if self.save_best_by == "val_acc"
                else val_loss
            )

            improved = (
                current_metric > self._best_metric
                if self.save_best_by == "val_acc"
                else current_metric < self._best_metric
            )

            if improved:
                self._best_metric = current_metric
                self.best_epoch = epoch
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self._no_improve = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "save_best_by": self.save_best_by,
                }, self.best_ckpt)

                print(
                    f"  [OK] {self.save_best_by} cai thien "
                    f"-> luu checkpoint (epoch {epoch})"
                )
            else:
                self._no_improve += 1

            if (
                self.patience is not None
                and self._no_improve >= self.patience
            ):
                print(
                    f"\n  Early stopping tai epoch {epoch} "
                    f"(patience={self.patience})"
                )
                break


        history_full = {
            **self.history,
            "best_epoch":   self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "save_best_by": self.save_best_by,
        }
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history_full, f, indent=2)


        total_min = (time.time() - t0) / 60
        print(f"\n{'='*55}")
        print(f"  Xong! Thoi gian: {total_min:.1f} phut")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}% tai epoch {self.best_epoch}")
        print(f"  Checkpoint: {self.best_ckpt}")
        print(f"{'='*55}\n")

        return history_full

    def load_best(self) -> None:
        """Load best checkpoint weights vào model."""
        ckpt = torch.load(self.best_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        val_acc  = ckpt.get("val_acc",  ckpt.get("best_val_acc"))
        val_loss = ckpt.get("val_loss", ckpt.get("best_val_loss"))
        print(
            f"  Loaded checkpoint epoch {ckpt['epoch']}"
            f" (Val Acc: {val_acc:.2f}%  Val Loss: {val_loss:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI Script
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LeNet hoặc ResNet18 trên dataset chó-mèo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",        choices=["lenet", "resnet18", "alexnet", "vgg16"], required=True)
    parser.add_argument("--data-dir",     type=Path, default=ROOT / "datastet" / "raw")
    parser.add_argument("--output-dir",   type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs",       type=int,   default=30)    # Số epoch huấn luyện: mô hình học qua toàn bộ tập dữ liệu 30 lần
    parser.add_argument("--batch-size",   type=int,   default=64)    # Số mẫu dữ liệu xử lý trong mỗi batch
    parser.add_argument("--lr",           type=float, default=1e-3)  # Learning rate: tốc độ cập nhật trọng số của mô hình
    parser.add_argument("--weight-decay", type=float, default=1e-4) # Hệ số regularization để hạn chế overfitting
    parser.add_argument("--num-workers",  type=int,   default=4)    # Số tiến trình phụ dùng để tải và chuẩn bị dữ liệu
    parser.add_argument("--seed",         type=int,   default=42)   # Seed để tăng khả năng tái lập kết quả ngẫu nhiên
    parser.add_argument("--patience",     type=int,   default=None) # Số epoch cho phép không cải thiện trước khi dừng sớm (early stopping); None = không giới hạn
    parser.add_argument("--save-best-by", choices=["val_acc", "val_loss"], default="val_loss", help="Tiêu chí lưu checkpoint tốt nhất ('val_acc' hoặc 'val_loss')")
    return parser.parse_args()


def _build_model(model_name: str) -> nn.Module:
    if model_name == "lenet":
        from models.lenet import lenet
        return lenet(num_classes=2)
    if model_name == "alexnet":
        from models.alexnet import alexnet
        return alexnet(num_classes=2)
    if model_name == "vgg16":
        from models.vgg import vgg16
        return vgg16(num_classes=2)
    from models.resnet18 import resnet18
    return resnet18(num_classes=2)


def main() -> None:
    from data    import CatsDogsDataset, get_transforms
    from evaluate import plot_training_history
    from utils   import device_info, get_device, set_seed

    args = _parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"\nThiet bi : {device_info(device)}")
    print(f"Mo hinh  : {args.model.upper()}")
    print(f"Dataset  : {args.data_dir}")

    train_ds = CatsDogsDataset(args.data_dir, "train", get_transforms(args.model, "train"))
    val_ds   = CatsDogsDataset(args.data_dir, "val",   get_transforms(args.model, "val", augment=False))
    print(f"Train: {len(train_ds):,} anh  |  Val: {len(val_ds):,} anh")

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model    = _build_model(args.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham so  : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion,
        device=device, output_dir=args.output_dir / "metrics",
        model_name=args.model, scheduler=scheduler, patience=args.patience,
        save_best_by=args.save_best_by,
    )
    history = trainer.fit(train_loader, val_loader, num_epochs=args.epochs)

    plot_training_history(
        history_path=trainer.history_path,
        output_path=args.output_dir / "figures" / f"{args.model}_training_history.png",
        model_name=args.model,
    )
    print(f"\nTraining {args.model.upper()} hoan tat!")
    print(f"Best Val Acc: {history['best_val_acc']:.2f}% (epoch {history['best_epoch']})")


if __name__ == "__main__":
    main()
