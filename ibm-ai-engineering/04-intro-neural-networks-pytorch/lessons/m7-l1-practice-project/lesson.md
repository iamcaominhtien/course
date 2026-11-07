# Module 7 · Bài 1: Practice Project, mạng nơ-ron phân loại ung thư vú

> Nguồn: Module "Final Project, Final Quiz, and Course Wrap-Up", mục Practice Project (1 lab không chấm điểm). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian: đọc ~20 phút, làm lab ~1 giờ 30 phút. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Dựng một quy trình phân loại hoàn chỉnh trên dữ liệu thật: tải, cân bằng, tiền xử lý, chia tập, chuẩn hóa, DataLoader.
2. Xây mạng nơ-ron **một lớp ẩn** với `ReLU` và huấn luyện bằng `CrossEntropyLoss` + `Adam`.
3. Theo dõi train loss và test loss để nhận biết overfitting.
4. Thử thay đổi optimizer, kích thước mạng và bộ dữ liệu (Iris, 3 lớp).

Đây là bước chuẩn bị cho Final Project: mọi thành phần của nó bạn đã gặp ở đây hoặc ở các module trước.

## 1. Bộ dữ liệu
**Breast Cancer Wisconsin (Diagnostic)** là bộ dữ liệu kinh điển: 569 mẫu, mỗi mẫu có 30 đặc trưng số mô tả nhân tế bào (bán kính, kết cấu, chu vi...), và nhãn **B** (benign, lành tính) hoặc **M** (malignant, ác tính). Phân bố gốc: 357 B, 212 M, tức **mất cân bằng**.

Khóa lấy dữ liệu bằng thư viện `ucimlrepo` (kho UCI): `fetch_ucirepo(id=17)` rồi đọc `.data.features` và `.data.targets`.

### Cân bằng lớp
Khi một lớp áp đảo, mô hình có thể đạt độ chính xác cao chỉ bằng cách luôn đoán lớp đông hơn. Cách đơn giản trong bài này: lấy ngẫu nhiên **200 mẫu mỗi lớp** (`DataFrame.sample(n=200, random_state=42)`), tổng 400 mẫu, rồi nối lại.

## 2. Tiền xử lý
1. **Nhãn thành số:** `B → 0`, `M → 1` (`Series.map`).
2. **Chia 80/20** bằng `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`. `stratify=y` giữ tỉ lệ lớp ở cả hai tập: ở đây test có 40 + 40.
3. **Chuẩn hóa** bằng `StandardScaler`: mỗi cột về trung bình 0, độ lệch chuẩn 1. Quy tắc quan trọng: `fit_transform` trên **train**, chỉ `transform` trên **test**. Nếu `fit` trên test, thông tin từ dữ liệu "chưa thấy" rò rỉ vào bước chuẩn bị và kết quả đánh giá sẽ đẹp hơn thực tế (rò rỉ dữ liệu, data leakage). Kiểm tra: sau chuẩn hóa, trung bình từng cột của train bằng 0, còn của test thì **không** đúng bằng 0.
4. **Tensor:** đặc trưng `float32`; nhãn `long` (số nguyên 64 bit), vì `CrossEntropyLoss` mong nhãn là chỉ số lớp kiểu `long`.
5. **`TensorDataset` + `DataLoader`:** `TensorDataset(X, y)` là một `Dataset` dựng sẵn ghép hai tensor theo hàng; `DataLoader(..., batch_size=2, shuffle=True)` cho train, `shuffle=False` cho test.

## 3. Mạng nơ-ron
```python
class ClassificationNet(nn.Module):
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super().__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)
    def forward(self, x):
        x = torch.relu(self.fc1(x))      # lớp ẩn + hàm kích hoạt
        return self.fc2(x)               # hai điểm số, chưa phải xác suất
```
- **Lớp ẩn + ReLU** là điểm mới so với logistic regression: mạng học được ranh giới phi tuyến.
- Đầu ra có **hai** điểm số (một cho mỗi lớp), không có sigmoid ở cuối.
- Số tham số: `30·64 + 64` (lớp 1) `+ 64·2 + 2` (lớp 2) = 2114.

### `CrossEntropyLoss` so với `BCELoss`
Bài trước dùng một đầu ra + sigmoid + `BCELoss`. Ở đây dùng **hai đầu ra** + `nn.CrossEntropyLoss`:
- nhận **điểm số thô** (logit), tự áp dụng softmax bên trong, nên **không** thêm sigmoid/softmax vào `forward`;
- nhãn là chỉ số lớp (`0` hoặc `1`, kiểu `long`);
- tổng quát cho **bất kỳ số lớp** nào: chỉ cần đổi `output_units` (ví dụ 3 cho Iris).
Hai cách tương đương về bản chất (nhị phân); `CrossEntropyLoss` là chuẩn cho đa lớp.

Mô hình chưa huấn luyện dự đoán gần như 50/50 nên loss khoảng `−ln(0.5) ≈ 0.693`. Đó là mốc để kiểm tra: nếu loss lúc đầu chênh quá xa thì có thể sai shape hoặc sai nhãn.

### Optimizer
`optim.Adam(model.parameters(), lr=0.001)`: tự điều chỉnh bước theo từng tham số (xem bảng ở Module 6, Bài 2).

## 4. Vòng lặp huấn luyện hai pha
Mỗi epoch:
- **Pha train:** `model.train()`; duyệt lô: `zero_grad`, forward, loss, `backward`, `step`; loss epoch = trung bình loss các lô.
- **Pha đánh giá:** `model.eval()` và `torch.no_grad()`; duyệt `test_loader`, chỉ tính loss, không cập nhật.

Lưu hai list `train_losses` và `test_losses`, vẽ chung một đồ thị. **Cách đọc:** hai đường giảm cùng nhau và gần nhau là tốt; train tiếp tục giảm còn test đi ngang rồi tăng là overfitting (nhắc lại Module 4).

Độ chính xác: chỉ số lớp dự đoán là điểm số lớn nhất, `output.argmax(dim=1)`, rồi so với nhãn thật. Với cấu hình mặc định và 10 epoch, bạn sẽ thấy độ chính xác test trên 93%.

## 5. Ba bài luyện tập
1. **Đổi optimizer** sang `SGD` với `momentum=0.9`, `weight_decay=0.0001`. *Momentum* tích lũy hướng đi của các bước trước để vượt qua dao động; *weight decay* là L2 regularization.
2. **Đổi số nơ-ron ở lớp ẩn** (ví dụ 16). Số tham số giảm từ 2114 xuống 530; mô hình nhỏ hơn vẫn có thể đủ cho bài toán này.
3. **Bộ dữ liệu Iris:** 150 mẫu, 4 đặc trưng, **3 lớp**. Cùng quy trình, chỉ khác `input_units=4`, `output_units=3` (khóa gợi ý đọc từ `sklearn.datasets.load_iris`).

## Tóm tắt
- Quy trình: tải, cân bằng, nhãn số, chia stratify, chuẩn hóa chỉ từ train, tensor (`float32`, nhãn `long`), DataLoader.
- Mạng một lớp ẩn với ReLU; `CrossEntropyLoss` nhận logit và nhãn chỉ số lớp; Adam.
- Vòng lặp hai pha `train()` / `eval()`; so sánh train loss và test loss để nhận biết overfitting.
- Rò rỉ dữ liệu: không bao giờ `fit` bất cứ thứ gì trên tập test.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
