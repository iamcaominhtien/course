# Module 3 · Bài 1: Hồi quy tuyến tính: dự đoán và hàm loss

> Nguồn: Module "Applying Linear Regression and Gradient Descent", Lesson 1 (4 video, 1 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~30 phút. Làm lab: ~1 giờ. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Mô tả bài toán hồi quy tuyến tính 1 chiều: dữ liệu, mô hình `ŷ = w·x + b`, nhiễu.
2. Viết bước dự đoán (forward pass) bằng tensor, bằng `nn.Linear`, và bằng một lớp `nn.Module` tự viết.
3. Đọc tham số của mô hình (`parameters()`, `state_dict()`).
4. Giải thích hàm loss, đặc biệt MSE, và vì sao loss là hàm của các tham số.

## 1. Bài toán
Có một tập dữ liệu gồm các cặp (đầu vào, giá trị cần dự đoán): `(x₁, y₁), (x₂, y₂), …, (xₙ, yₙ)`. Ví dụ: diện tích nhà → giá nhà, giờ học → điểm thi. Đây là **học có giám sát** (supervised learning): mỗi ví dụ huấn luyện có sẵn đáp án.

Khi chỉ có **một** đặc trưng đầu vào và quan hệ gần như đường thẳng, ta dùng **hồi quy tuyến tính đơn** (simple linear regression):

> **ŷ = w · x + b**

- `x`: đầu vào (còn gọi là biến độc lập, predictor).
- `ŷ` ("y mũ"): giá trị mô hình dự đoán. `y` là giá trị thật.
- `w` (weight): độ dốc của đường thẳng, cho biết `x` ảnh hưởng đến kết quả bao nhiêu.
- `b` (bias): hệ số chặn, dịch đường thẳng lên xuống.

**Huấn luyện** là tìm `w` và `b` sao cho đường thẳng khớp dữ liệu nhất.

### Nhiễu (noise)
Dữ liệu thật hiếm khi nằm đúng trên một đường thẳng, vì có sai số đo và những yếu tố ta không biết. Phần lệch đó gọi là **nhiễu**, thường được giả định tuân theo phân phối Gauss (chuẩn): phần lớn độ lệch nhỏ và quanh 0, độ lệch lớn hiếm hơn. Nhiễu càng lớn thì điểm dữ liệu càng tản xa đường thẳng thật. Mục tiêu không đổi: bắt được xu hướng chung, đừng cố khớp từng điểm.

## 2. Bước dự đoán (forward pass)
Đưa `x` qua mô hình để ra `ŷ` gọi là **dự đoán**, hay **forward pass**. Làm thủ công bằng tensor:

```python
import torch
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

def forward(x):
    return w * x + b

forward(torch.tensor([[1.0]]))              # tensor([[1.]])        vì 2·1 − 1 = 1
forward(torch.tensor([[1.0], [2.0]]))       # tensor([[1.], [3.]])
```
Hai điều cần để ý:
- **Mỗi hàng là một mẫu.** Với `x` shape `(2, 1)` (2 mẫu, 1 đặc trưng), kết quả cũng `(2, 1)`: một dự đoán cho mỗi mẫu. Cùng một công thức áp lên nhiều mẫu một lúc nhờ broadcasting.
- `requires_grad=True` để PyTorch tính được gradient của `w` và `b` ở bài sau.

## 3. `nn.Linear`: lớp tuyến tính dựng sẵn
Không cần tự viết `w * x + b`; PyTorch có sẵn lớp `torch.nn.Linear`:

```python
from torch.nn import Linear
torch.manual_seed(1)                       # tham số khởi tạo ngẫu nhiên → cố định seed để tái lập
model = Linear(in_features=1, out_features=1, bias=True)
list(model.parameters())                   # w ≈ 0.5153, b ≈ −0.4414 (với seed 1)
model(torch.tensor([[1.0], [2.0]]))        # gọi thẳng model như một hàm
```
- `in_features` là số đặc trưng đầu vào, `out_features` là số đầu ra. Lớp tự tạo và quản lý `weight`, `bias`.
- Gọi `model(x)` thực chất gọi bước forward.
- `model.state_dict()` trả về dictionary tên → tensor tham số (`weight`, `bias`). Khi lưu mô hình, ta lưu đúng thứ này. Có thể đọc riêng `model.weight`, `model.bias`.
- Với `Linear(1, 1)`, `weight` có shape `(1, 1)` và `bias` shape `(1,)`.

## 4. Mô hình tự viết với `nn.Module`
Mô hình thực tế gồm nhiều lớp ghép lại. Cách chuẩn trong PyTorch: viết một lớp kế thừa `nn.Module`.

```python
from torch import nn

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()                              # bắt buộc: khởi tạo lớp cha
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):                               # định nghĩa cách tính dự đoán
        return self.linear(x)

model = LR(1, 1)
model(torch.tensor([[1.0], [2.0]]))     # tự gọi forward
model.state_dict()                      # key: 'linear.weight', 'linear.bias'
```
- `__init__` khai báo các thành phần (lớp con); `forward` nói cách dùng chúng để ra dự đoán.
- Không gọi `model.forward(x)` trực tiếp; gọi `model(x)` để PyTorch xử lý các việc phụ (hook...).
- Vì kế thừa `nn.Module` nên `parameters()` và `state_dict()` tự có; tên tham số có tiền tố tên thuộc tính (`linear.weight`).

## 5. Huấn luyện nghĩa là gì?
**Huấn luyện** = học tham số của mô hình từ dữ liệu để dự đoán tốt. Để biết dự đoán "tốt" hay "dở", ta cần một thước đo. Đó là **hàm loss**.

## 6. Hàm loss
**Loss function** so sánh giá trị dự đoán với giá trị thật và đổi **sai số dự đoán** thành **một con số**: loss nhỏ là dự đoán tốt, loss lớn là dự đoán tệ.

Với hồi quy, hàm loss phổ biến là **Mean Squared Error (MSE)**, sai số bình phương trung bình:

> **MSE = (1/N) · Σ (ŷᵢ − yᵢ)²**

Vì sao bình phương?
1. Sai số dương và âm không triệt tiêu nhau.
2. Sai số lớn bị phạt nặng hơn sai số nhỏ, nên mô hình chú ý giảm các sai lệch lớn.

```python
import torch.nn as nn
criterion = nn.MSELoss()
criterion(torch.tensor([1., 2., 3.]), torch.tensor([1., 1., 5.]))   # tensor(1.6667)
# kiểm tra tay: ((1-1)² + (2-1)² + (3-5)²) / 3 = (0 + 1 + 4) / 3 = 1.6667
```
Hoặc tự viết: `torch.mean((yhat - y) ** 2)`.

### Loss là hàm của tham số
Dự đoán phụ thuộc `w` và `b`, nên loss cũng phụ thuộc `w` và `b`. Đổi tham số thì dự đoán đổi, loss đổi theo. Nhìn theo cách này, **huấn luyện là bài toán tối ưu**: tìm `w, b` làm loss nhỏ nhất. Khi loss nhỏ nhất, đường thẳng là "mô hình tuyến tính khớp nhất". Cách tìm (gradient descent) là nội dung bài sau.

## Tóm tắt
- Mô hình `ŷ = w·x + b`; mỗi hàng của `x` là một mẫu, mỗi mẫu ra một dự đoán.
- Dự đoán = forward pass. Dựng bằng tensor, `nn.Linear`, hoặc lớp con của `nn.Module`.
- `parameters()` và `state_dict()` cho thấy tham số mà quá trình huấn luyện sẽ cập nhật.
- Loss đo sai số dự đoán; MSE = trung bình bình phương sai số. Loss là hàm của tham số, nên huấn luyện là đi tìm tham số làm loss nhỏ nhất.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
