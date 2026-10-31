# CNN phân loại mèo và chó

Khung dự án phục vụ chuẩn bị dữ liệu và phát triển các mạng nơ-ron tích chập
(CNN) để phân loại ảnh mèo và chó.

# Hiện đã hoàn thiện
Dataset ảnh tại datastet/raw, chia thành train, val, test; mỗi tập có thư mục cats và dogs.
Train: 8.000 mèo, 7.989 chó
Val: 2.000 mèo, 1.998 chó
Test: 2.500 mèo, 2.500 chó

Bộ đọc ảnh dataset.py: đọc ảnh theo thư mục lớp, đổi sang RGB, gán cats = 0, dogs = 1.
Transform transforms.py: hỗ trợ kích thước đầu vào cho LeNet và ResNet18; augmentation chỉ áp dụng cho tập train.
Validator validate_data.py kiểm tra cấu trúc thư mục, file ảnh lỗi và ảnh trùng giữa các tập.

# Hiện chưa hoàn thiện

Để project có thể chạy hoàn chỉnh và phân loại ảnh chó/mèo, cần triển khai
thêm các phần dưới đây. Đây là kế hoạch các bước tiếp theo; những chức năng
này hiện chưa được cài đặt.

## Các bước cần triển khai

1. **Hoàn thiện hai mô hình phân loại**
   - Viết kiến trúc LeNet trong `models/base_cnn.py` và ResNet18 trong
     `models/resnet18.py`.
   - Cả hai mô hình cần nhận ảnh RGB 3 kênh và trả về điểm số cho đúng 2 lớp:
     `cats` và `dogs`.
   - Với ResNet18, có thể hỗ trợ tùy chọn dùng trọng số pretrained để so sánh
     transfer learning với huấn luyện từ đầu. Khi dùng pretrained, cần bảo đảm
     tiền xử lý ảnh khớp với bộ trọng số.

2. **Tạo DataLoader cho train, validation và test**
   - Dùng `CatsDogsDataset` cùng transform tương ứng của từng mô hình.
   - Gom ảnh thành batch bằng `DataLoader`; cho phép đặt batch size, trộn dữ
     liệu train và cấu hình số worker.
   - Chỉ augmentation tập train. Dùng validation để theo dõi/chọn mô hình và
     chỉ dùng test để đánh giá cuối cùng.

3. **Viết quy trình huấn luyện**
   - Hoàn thiện `src/cats_dogs/training/engine.py` và `trainer.py`, cùng
     `scripts/train.py`.
   - Hỗ trợ loss phân loại nhị phân hoặc cross-entropy hai lớp, optimizer,
     số epoch, thiết bị CPU/GPU và seed để có thể tái lập kết quả.
   - Ghi nhận loss và chỉ số trên train/validation theo từng epoch; lưu
     checkpoint tốt nhất dựa trên validation, không dựa trên test.

4. **Cài đặt đánh giá**
   - Hoàn thiện `src/cats_dogs/evaluation/metrics.py` và
     `scripts/evaluate.py`.
   - Tính accuracy, precision, recall, F1-score và confusion matrix trên tập
     test; lưu kết quả vào `outputs/metrics/`.

5. **Cài đặt dự đoán ảnh mới**
   - Hoàn thiện `src/cats_dogs/inference/predictor.py` và
     `scripts/predict.py`.
   - Nạp checkpoint đã lưu, áp dụng đúng transform của mô hình, rồi trả về
     nhãn `cat`/`dog` cùng xác suất dự đoán. Có thể lưu kết quả vào
     `outputs/predictions/`.

6. **Bổ sung kiểm thử cho mô hình và quy trình**
   - Trong `tests/test_model.py`, kiểm tra mỗi mô hình nhận batch RGB và trả
     đầu ra có 2 lớp với đúng kích thước.
   - Trong `tests/test_training.py`, kiểm tra một bước huấn luyện nhỏ trên dữ
     liệu giả, việc lưu/nạp checkpoint và xử lý thiết bị.
   - Kiểm thử đánh giá và dự đoán để bảo đảm cùng một ánh xạ lớp được dùng
     xuyên suốt (`cats = 0`, `dogs = 1`).

7. **Ghi lại cách cài đặt và sử dụng**
   - Tạo `requirements.txt` hoặc cấu hình phụ thuộc phù hợp, bao gồm phiên bản
     tương thích của PyTorch, torchvision, Pillow và các thư viện đánh giá.
   - Bổ sung hướng dẫn tạo môi trường, chạy kiểm tra dữ liệu, huấn luyện từng
     mô hình, đánh giá checkpoint và dự đoán một ảnh mới.

## Điều kiện để xem là chạy hoàn chỉnh

- Validator dữ liệu và toàn bộ kiểm thử đều chạy thành công.
- Lệnh huấn luyện có thể chạy trên một cấu hình nhỏ để xác nhận pipeline
  hoạt động từ lúc đọc ảnh đến lúc lưu checkpoint; việc xác nhận này không
  thay thế đánh giá chất lượng trên tập test.
- Lệnh đánh giá nạp được checkpoint và tạo đủ chỉ số cho tập test.
- Lệnh dự đoán xử lý được ảnh mới và trả về một trong hai nhãn mèo hoặc chó.
- LeNet và ResNet18 được đánh giá trên cùng các split, cùng ánh xạ nhãn; không
  dùng tập test để chọn cấu hình hay điều chỉnh mô hình.

## Cấu trúc dự án

- `datastet/raw/`: ảnh đã tải về, được sắp xếp theo `train/{cats,dogs}`,
  `val/{cats,dogs}` và `test/{cats,dogs}`. Tập validation được tách xác định
  từ tập train bằng seed 42 (20% mỗi lớp); tập test được giữ nguyên. Hai ảnh
  trong train trùng hoàn toàn với ảnh trong test được lưu tại
  `datastet/excluded_duplicates/` và không đưa vào các tập dữ liệu.
- `src/cats_dogs/data/`: bộ đọc ảnh, phép biến đổi ảnh theo mô hình và công cụ
  kiểm tra tính toàn vẹn của dữ liệu.
- `models/`: nơi định nghĩa các mô hình.
- `src/cats_dogs/`: các thành phần xử lý dữ liệu, huấn luyện, đánh giá, suy
  luận và tiện ích.
- `scripts/`: các tập lệnh dòng lệnh.
- `notebooks/`: sổ tay khám phá dữ liệu.
- `tests/`: các bài kiểm thử của dự án.
- `outputs/`: hình ảnh, kết quả dự đoán và chỉ số được tạo ra.

## Chuẩn bị dữ liệu cho LeNet và ResNet18

Dự án này chỉ chuẩn bị và kiểm tra dữ liệu, không huấn luyện hai mô hình.
Ảnh được đọc trực tiếp từ `datastet/raw/`; không tạo bản sao đã đổi kích thước
trên ổ đĩa. Hai ảnh chó trùng từng byte giữa `train` và `test` đã được chuyển
từ `train` sang `datastet/excluded_duplicates/` để tránh rò rỉ dữ liệu kiểm
tra. Bộ đọc lấy nhãn từ thư mục theo ánh xạ cố định (`cats = 0`, `dogs = 1`),
chuyển ảnh sang RGB và áp dụng phép biến đổi khi tải từng ảnh.

- LeNet nhận tensor RGB có kích thước `3 x 32 x 32`, được chuẩn hóa về
  khoảng `[-1, 1]`. Kiến trúc LeNet khi triển khai cần nhận đầu vào ba kênh.
- ResNet18 nhận tensor RGB kích thước `3 x 224 x 224`, sau khi đổi kích thước,
  cắt giữa ảnh và chuẩn hóa theo ImageNet.
- Mặc định, phép lật ngang ngẫu nhiên chỉ áp dụng cho tập train. Phép biến đổi
  của validation và test có tính xác định và không tăng cường dữ liệu.
- Dữ liệu train, validation và test được đọc từ các thư mục tương ứng; các
  tiện ích chuẩn bị dữ liệu không dùng tập test để huấn luyện.

Cài đặt PyTorch, torchvision và Pillow trong môi trường Python đang sử dụng.
Sau đó, chạy lệnh sau từ thư mục gốc dự án bằng PowerShell để kiểm tra tất cả
ảnh, thư mục lớp cần thiết và ảnh có nội dung giống hệt nhau giữa các tập:

```powershell
python .\scripts\validate_data.py
```

Để sử dụng dữ liệu trong một tập lệnh huấn luyện sau này, hãy tạo một dataset
cho mỗi tập:

```python
from cats_dogs.data import CatsDogsDataset, get_transforms

train_data = CatsDogsDataset(
    "datastet/raw", "train", get_transforms("lenet", "train")
)
val_data = CatsDogsDataset("datastet/raw", "val", get_transforms("lenet", "val"))
test_data = CatsDogsDataset("datastet/raw", "test", get_transforms("lenet", "test"))
```

Thiết lập `PYTHONPATH=src` (hoặc cài đặt gói của dự án) khi import `cats_dogs`
từ một tập lệnh riêng. Thay `"lenet"` bằng `"resnet18"` để sử dụng kích thước
đầu vào và cách chuẩn hóa tương ứng.

Chạy các bài kiểm thử chuẩn bị dữ liệu trong PowerShell bằng lệnh:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```
