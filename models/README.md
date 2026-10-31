# Mô Hình Phân Loại Ảnh Chó - Mèo (Cats vs Dogs)

Thư mục này chứa mã nguồn định nghĩa hai kiến trúc mạng nơ-ron tích chập (Convolutional Neural Network - CNN) được xây dựng hoàn chỉnh từ đầu (**train from scratch** bằng PyTorch), phục vụ bài toán phân loại nhị phân giữa ảnh **Mèo (nhãn 0)** và **Chó (nhãn 1)**.

---

## 1. Danh Sách Các File

| File | Mô hình | Kích thước ảnh đầu vào | Số tham số (Parameters) | Trọng số mặc định |
| :--- | :--- | :---: | :---: | :---: |
| [`lenet.py`](lenet.py) | **LeNet-5 (RGB)** | $3 \times 32 \times 32$ | $61,326$ | Khởi tạo ngẫu nhiên |
| [`resnet18.py`](resnet18.py) | **ResNet-18 (From Scratch)** | $3 \times 224 \times 224$ | $11,177,538$ | Khởi tạo Kaiming (He) |

---

## 2. Chi Tiết Kiến Trúc Từng Mô Hình

### 2.1. LeNet-5 (`lenet.py`)

* **Nguồn gốc:** Dựa trên kiến trúc LeNet-5 kinh điển của Yann LeCun et al. (1998) - mô hình đặt nền móng cho thị giác máy tính hiện đại với các tầng Conv, Pooling và Fully-Connected xen kẽ.
* **Tùy biến cho bài toán:**
  * Nhận ảnh màu **RGB 3 kênh** thay vì 1 kênh grayscale như nguyên bản.
  * Kích thước ảnh đầu vào: $32 \times 32$ pixels.
  * Tầng phân loại (Classifier) đưa về **2 classes** (`cats`, `dogs`).
  * Hàm kích hoạt sử dụng **ReLU** thay vì hàm Tanh/Sigmoid cổ điển.

#### Cấu trúc luồng xử lý (Data Flow):
```text
Input (N, 3, 32, 32)
   │
   ▼
[Conv2d (3 -> 6, k=5)] + ReLU ──> (N, 6, 28, 28)
   │
   ▼
[MaxPool2d (2x2, stride=2)]  ───> (N, 6, 14, 14)
   │
   ▼
[Conv2d (6 -> 16, k=5)] + ReLU ─> (N, 16, 10, 10)
   │
   ▼
[MaxPool2d (2x2, stride=2)]  ───> (N, 16, 5, 5)
   │
   ▼
[Flatten]                    ───> (N, 400)
   │
   ▼
[Linear (400 -> 120)] + ReLU ───> (N, 120)
   │
   ▼
[Linear (120 -> 84)] + ReLU  ───> (N, 84)
   │
   ▼
[Linear (84 -> 2)]           ───> Logits (N, 2)
```

#### Phân bố tham số:
* **Feature Extractor:**
  * `conv1`: $3 \times 6 \times 5 \times 5 + 6 = 456$ tham số.
  * `conv2`: $6 \times 16 \times 5 \times 5 + 16 = 2,416$ tham số.
* **Classifier:**
  * `fc1`: $400 \times 120 + 120 = 48,120$ tham số.
  * `fc2`: $120 \times 84 + 84 = 10,164$ tham số.
  * `fc3`: $84 \times 2 + 2 = 170$ tham số.
* **Tổng số tham số:** $61,326$ (~0.06M tham số).

---

### 2.2. ResNet-18 (`resnet18.py`)

* **Nguồn gốc:** Dựa trên bài báo kinh điển *"Deep Residual Learning for Image Recognition"* của Kaiming He et al. (CVPR 2016). Mô hình giải quyết triệt để vấn đề suy thoái chất lượng mạng sâu (degradation problem / vanishing gradients) nhờ cơ chế kết nối tắt (**residual shortcut / skip connection**).
* **Đặc điểm cài đặt trong dự án:**
  * Xây dựng từ đầu (**from scratch**), không nạp trọng số có sẵn từ torchvision.
  * Kích thước ảnh đầu vào: $224 \times 224$ pixels.
  * Đầu ra: 2 classes (`cats`, `dogs`).
  * Khởi tạo trọng số tối ưu: **Kaiming He Normal** cho tầng tích chập (`Conv2d`), cố định bias và hệ số tỉ lệ cho `BatchNorm2d`, chuẩn hóa phân phối thường cho tầng `Linear`.

#### Khối `BasicBlock`:
Mỗi khối Residual cơ bản gồm 2 tầng Conv $3 \times 3$, đi kèm Batch Normalization và hàm kích hoạt ReLU:
$$F(x) = \text{BN}_2(\text{Conv}_2(\text{ReLU}(\text{BN}_1(\text{Conv}_1(x)))))$$
Đầu ra của khối là:
$$y = \text{ReLU}(F(x) + \text{shortcut}(x))$$
* **Identity Shortcut:** Giữ nguyên $x$ khi kích thước không gian và số kênh không đổi.
* **Projection Shortcut:** Sử dụng tầng Conv $1 \times 1$ kết hợp BatchNorm để khớp số kênh hoặc hạ mẫu không gian khi $stride > 1$.

#### Cấu trúc luồng xử lý:
1. **Stem:**
   * `Conv2d (3 -> 64, k=7x7, stride=2, padding=3)` $\rightarrow (N, 64, 112, 112)$
   * `BatchNorm2d(64)` + `ReLU`
   * `MaxPool2d (3x3, stride=2, padding=1)` $\rightarrow (N, 64, 56, 56)$
2. **Residual Stages (4 stages, mỗi stage gồm 2 BasicBlock):**
   * **Layer 1:** 2 BasicBlocks, 64 kênh, stride = 1 $\rightarrow (N, 64, 56, 56)$
   * **Layer 2:** 2 BasicBlocks, 128 kênh, stride = 2 $\rightarrow (N, 128, 28, 28)$
   * **Layer 3:** 2 BasicBlocks, 256 kênh, stride = 2 $\rightarrow (N, 256, 14, 14)$
   * **Layer 4:** 2 BasicBlocks, 512 kênh, stride = 2 $\rightarrow (N, 512, 7, 7)$
3. **Head:**
   * `AdaptiveAvgPool2d((1, 1))` $\rightarrow (N, 512, 1, 1)$
   * `Flatten` $\rightarrow (N, 512)$
   * `Linear(512, 2)` $\rightarrow$ Logits $(N, 2)$

* **Tổng số tham số:** $11,177,538$ (~11.18M tham số).

---

## 3. Bảng So Sánh Hai Mô Hình

| Đặc trưng | LeNet-5 | ResNet-18 |
| :--- | :--- | :--- |
| **File triển khai** | [`lenet.py`](lenet.py) | [`resnet18.py`](resnet18.py) |
| **Kích thước ảnh vào** | $3 \times 32 \times 32$ | $3 \times 224 \times 224$ |
| **Phương pháp tiền xử lý** | Resize $32 \times 32$, chuẩn hóa $[-1, 1]$ | Resize 256, Crop 224, chuẩn hóa ImageNet |
| **Số tham số (Params)** | $61,326$ (~0.06M) | $11,177,538$ (~11.18M) |
| **Dung lượng file weights (.pth)** | ~748 KB | ~134 MB |
| **Cơ chế cốt lõi** | Conv thuần túy + MaxPool + FC | Residual Blocks (Skip connections) + BatchNorm + Global Average Pooling |
| **Độ sâu** | 5 tầng có trọng số (2 Conv + 3 FC) | 18 tầng có trọng số (1 Stem Conv + 16 Residual Conv + 1 FC) |
| **Test Accuracy (Thực nghiệm)** | **76.74%** | **95.06%** |
| **Macro F1-score (Thực nghiệm)** | **76.71%** | **95.06%** |
| **Thời gian train / inference** | Cực nhanh, tiêu tốn ít RAM/VRAM | Tốn nhiều thời gian và tài nguyên hơn |
| **Khả năng nắm bắt đặc trưng** | Giới hạn ở hình thái tổng thể độ phân giải thấp | Học được đặc trưng phân cấp sâu, chi tiết vân lông, mắt, tai,... |

---

## 4. Hướng Dẫn Sử Dụng Code

### 4.1. Khởi tạo và sử dụng mô hình trong Python

```python
import torch
from models.lenet import lenet
from models.resnet18 import resnet18

# 1. Khởi tạo LeNet (Input: 3 x 32 x 32)
model_lenet = lenet(num_classes=2)
dummy_input_lenet = torch.randn(4, 3, 32, 32)
output_lenet = model_lenet(dummy_input_lenet)
print("LeNet output shape:", output_lenet.shape)      # torch.Size([4, 2])

# 2. Khởi tạo ResNet-18 (Input: 3 x 224 x 224)
model_resnet = resnet18(num_classes=2)
dummy_input_resnet = torch.randn(4, 3, 224, 224)
output_resnet = model_resnet(dummy_input_resnet)
print("ResNet-18 output shape:", output_resnet.shape) # torch.Size([4, 2])
```

### 4.2. Huấn luyện mô hình từ CLI

Dự án hỗ trợ train cả hai mô hình qua file `train.py` tại thư mục gốc:

```bash
# Huấn luyện LeNet
python train.py --model lenet --epochs 30 --batch-size 64 --lr 0.001

# Huấn luyện ResNet-18
python train.py --model resnet18 --epochs 30 --batch-size 32 --lr 0.001
```

### 4.3. Nạp trọng số checkpoint đã huấn luyện

```python
import torch
from models.resnet18 import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=2).to(device)

checkpoint = torch.load("outputs/metrics/resnet18_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Đã nạp checkpoint (Best Val Acc: {checkpoint['best_val_acc']:.2f}%)")
```

---

## 5. Nhận Xét & Kết Luận

1. **LeNet-5:**
   * Đóng vai trò là mô hình **baseline** để kiểm chứng luồng huấn luyện dữ liệu.
   * Thích hợp cho các bài toán nhẹ hoặc thiết bị nhúng có tài nguyên cực kỳ hạn hẹp. Tuy nhiên, việc nén ảnh xuống $32 \times 32$ làm mất nhiều chi tiết nhận dạng giữa mèo và chó, dẫn đến giới hạn độ chính xác quanh mức ~76.7%.
2. **ResNet-18:**
   * Đóng vai trò là mô hình **nâng cao (SOTA baseline)** trong phân loại ảnh.
   * Giữ được độ phân giải ảnh $224 \times 224$, cho phép mạng học các đặc trưng tinh vi. Kết nối tắt (residual connections) và Batch Normalization giúp mô hình hội tụ tốt mà không bị biến mất gradient, đạt độ chính xác vượt trội trên tập kiểm thử (~95.1%).

