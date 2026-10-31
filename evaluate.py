"""Đánh giá và dự đoán kết quả mô hình phân loại chó-mèo.

Gộp metrics (accuracy, confusion matrix) + predictor (predict ảnh đơn / batch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

CLASS_NAMES = ["cats", "dogs"]


# ─── Thu thập predictions ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thu thập toàn bộ predictions trên DataLoader.

    Returns:
        (all_labels, all_preds, all_probs) — shape (N,), (N,), (N, C)
    """
    model.eval()
    model.to(device)
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images: Tensor = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu()
        preds = probs.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return (
        np.array(all_labels, dtype=np.int64),
        np.array(all_preds,  dtype=np.int64),
        np.array(all_probs,  dtype=np.float32),
    )


# ─── Tính metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Tính Accuracy, Precision, Recall, F1, Confusion Matrix (không cần sklearn)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    accuracy = float(np.diag(cm).sum()) / len(labels) * 100
    per_class: dict = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec * 100, 2),
            "recall":    round(rec  * 100, 2),
            "f1":        round(f1   * 100, 2),
            "support":   int(cm[i, :].sum()),
        }

    return {
        "accuracy":        round(accuracy, 4),
        "macro_precision": round(float(np.mean([v["precision"] for v in per_class.values()])), 4),
        "macro_recall":    round(float(np.mean([v["recall"]    for v in per_class.values()])), 4),
        "macro_f1":        round(float(np.mean([v["f1"]        for v in per_class.values()])), 4),
        "per_class":       per_class,
        "confusion_matrix": cm.tolist(),
        "class_names":     class_names,
        "n_samples":       int(len(labels)),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Lưu metrics ra JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics → {path}")


# ─── Vẽ biểu đồ ──────────────────────────────────────────────────────────────

def plot_training_history(history_path: str | Path, output_path: str | Path, model_name: str = "model") -> None:
    """Vẽ biểu đồ loss & accuracy theo epoch từ file history JSON."""
    with open(history_path, encoding="utf-8") as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training History — {model_name.upper()}", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, h["train_loss"], "b-o", markersize=3, label="Train")
    axes[0].plot(epochs, h["val_loss"],   "r-o", markersize=3, label="Val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss theo Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, h["train_acc"], "b-o", markersize=3, label="Train")
    axes[1].plot(epochs, h["val_acc"],   "r-o", markersize=3, label="Val")
    if h.get("best_epoch", 0) > 0:
        axes[1].axvline(h["best_epoch"], color="green", linestyle="--", alpha=0.7,
                        label=f"Best Epoch {h['best_epoch']} ({h['best_val_acc']:.2f}%)")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy theo Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Biểu đồ training → {out}")


def plot_confusion_matrix(cm: list, class_names: list[str], output_path: str | Path, model_name: str = "model") -> None:
    """Vẽ confusion matrix dạng heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix — {model_name.upper()}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm_arr.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_arr[i,j]:,}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm_arr[i, j] > thresh else "black")
    plt.tight_layout()
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Confusion matrix → {out}")


def compare_models_plot(lenet_hist_path: str | Path, resnet_hist_path: str | Path, output_path: str | Path) -> None:
    """Vẽ biểu đồ so sánh Val Accuracy và Val Loss của LeNet vs ResNet18."""
    with open(lenet_hist_path,  encoding="utf-8") as f: lh = json.load(f)
    with open(resnet_hist_path, encoding="utf-8") as f: rh = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LeNet vs ResNet18 (Train from Scratch)", fontsize=14, fontweight="bold")

    axes[0].plot(range(1, len(lh["val_acc"])+1), lh["val_acc"], "b-", lw=2, label="LeNet")
    axes[0].plot(range(1, len(rh["val_acc"])+1), rh["val_acc"], "r-", lw=2, label="ResNet18")
    axes[0].set(xlabel="Epoch", ylabel="Val Accuracy (%)", title="Validation Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, len(lh["val_loss"])+1), lh["val_loss"], "b-", lw=2, label="LeNet")
    axes[1].plot(range(1, len(rh["val_loss"])+1), rh["val_loss"], "r-", lw=2, label="ResNet18")
    axes[1].set(xlabel="Epoch", ylabel="Val Loss", title="Validation Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  So sánh mô hình → {out}")


def compare_all_models_plot(
    history_paths: dict[str, str | Path],
    output_path: str | Path,
) -> None:
    """Vẽ biểu đồ so sánh Val Accuracy và Val Loss của tất cả các mô hình.

    Args:
        history_paths: Dict ánh xạ tên model → đường dẫn file history JSON.
                       VD: {"lenet": "...", "resnet18": "...", "alexnet": "...", "vgg16": "..."}
        output_path:   Đường dẫn lưu ảnh biểu đồ.
    """
    COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

    histories: dict[str, dict] = {}
    for name, path in history_paths.items():
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                histories[name] = json.load(f)
        else:
            print(f"  [WARN] Không tìm thấy history: {path} — bỏ qua {name}")

    if not histories:
        print("  [ERROR] Không có history nào để vẽ.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title_models = " vs ".join(k.upper() for k in histories)
    fig.suptitle(f"{title_models} — So sánh (Train from Scratch)", fontsize=13, fontweight="bold")

    for (name, h), color in zip(histories.items(), COLORS):
        epochs = range(1, len(h["val_acc"]) + 1)
        label = f"{name.upper()} (best {h.get('best_val_acc', 0):.1f}%)"
        axes[0].plot(epochs, h["val_acc"],  color=color, lw=2, label=label)
        axes[1].plot(epochs, h["val_loss"], color=color, lw=2, label=name.upper())

    axes[0].set(xlabel="Epoch", ylabel="Val Accuracy (%)", title="Validation Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set(xlabel="Epoch", ylabel="Val Loss", title="Validation Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  So sánh tất cả mô hình → {out}")


# ─── Predict ảnh đơn lẻ ──────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    transform: transforms.Compose,
    device: torch.device,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """Dự đoán nhãn cho một ảnh.

    Returns:
        Dict gồm predicted_class, confidence, prob_cats, prob_dogs.
    """
    model.eval()
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    idx = int(probs.argmax())
    return {
        "predicted_class": class_names[idx],
        "confidence":      round(float(probs[idx]) * 100, 2),
        "prob_cats":       round(float(probs[0]) * 100, 2),
        "prob_dogs":       round(float(probs[1]) * 100, 2),
    }
