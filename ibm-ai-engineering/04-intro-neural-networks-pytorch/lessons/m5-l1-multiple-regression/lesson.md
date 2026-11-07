# Module 5 · Bài 1: Hồi quy tuyến tính nhiều đầu vào và nhiều đầu ra

> Nguồn: Module "Extending Linear Regression to Multiple Inputs and Outputs" (5 video, 4 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~35 phút. Làm lab: ~1 giờ 30 phút. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Viết mô hình hồi quy nhiều đầu vào `ŷ = w·x + b` và theo dõi **shape** của tensor trong phép nhân ma trận.
2. Dùng `nn.Linear(in_features, out_features)` và lớp `nn.Module` tự viết cho trường hợp nhiều đầu vào, nhiều đầu ra.
3. Huấn luyện mô hình nhiều đầu vào/đầu ra với vòng lặp mini-batch quen thuộc.
4. Đọc đúng shape của `weight` và `bias` trong `nn.Linear`.
5. Hiểu vì sao `nn.Linear` là viên gạch cơ bản của mạng nơ-ron, và nắm vài xu hướng hiện đại quanh PyTorch.

## 1. Nhiều đầu vào
Bài toán thực tế hiếm khi chỉ có một đặc trưng. Giá nhà phụ thuộc diện tích, số phòng, vị trí... Với `d` đặc trưng `x₁ … x_d`:

> **ŷ = w₁x₁ + w₂x₂ + … + w_d·x_d + b = w · x + b**

Có **`d` trọng số và 1 bias**: 2 đặc trưng → 2 trọng số + 1 bias, 3 đặc trưng → 3 trọng số + 1 bias. Dạng vector: `w · x` là tích vô hướng (bài 1 của Module 1).

Cost vẫn là MSE; huấn luyện vẫn là gradient descent, chỉ khác là `w` giờ là một **vector** tham số:
> `w ← w − lr · ∂cost/∂w` (vector), `b ← b − lr · ∂cost/∂b`.

### Shape, shape, shape
Quy tắc nhân ma trận từ Module 1: `(m, n) @ (n, p) → (m, p)`. Với **một** mẫu có `d` đặc trưng:
- `x` shape `(1, d)`, `w` shape `(d, 1)` ⇒ `x @ w` shape `(1, 1)`, một dự đoán.

Với **nhiều** mẫu, xếp thành ma trận `X` shape `(N, d)`: mỗi **hàng** là một mẫu, mỗi **cột** là một đặc trưng. Khi đó `X @ w` shape `(N, 1)`: mỗi hàng một dự đoán.

```python
w = torch.tensor([[2.0], [3.0]], requires_grad=True)   # (2, 1)
b = torch.tensor([[1.0]], requires_grad=True)
def forward(x):
    return torch.mm(x, w) + b                          # matrix multiply + bias

forward(torch.tensor([[1.0, 2.0]]))                    # 1·2 + 2·3 + 1 = [[9.]]
X = torch.tensor([[1., 1.], [1., 2.], [1., 3.]])       # (3, 2): 3 mẫu, 2 đặc trưng
forward(X)                                             # [[6.], [9.], [12.]]
```
Điều kiện: **số cột của `x` = số hàng của `w`** (= số đặc trưng). Sai điều này là lỗi shape phổ biến nhất.

## 2. `nn.Linear` cho nhiều đầu vào
```python
from torch import nn
model = nn.Linear(in_features=2, out_features=1)
model(X)                      # X: (3, 2)  →  kết quả (3, 1)
```
- `in_features` = số đặc trưng đầu vào; `out_features` = số giá trị đầu ra.
- Lớp tự khởi tạo trọng số và bias. Đọc bằng `model.weight`, `model.bias`, `model.parameters()`, `model.state_dict()`.
- **Lưu ý shape:** PyTorch lưu `weight` dạng **`(out_features, in_features)`**, tức chuyển vị so với ma trận `W` viết trong công thức. `nn.Linear(2, 1).weight` có shape `(1, 2)`, và lớp tính `x @ weightᵀ + bias`. Đừng nhầm khi so với công thức tay.

Mô hình tự viết cũng giống Module 3, chỉ đổi kích thước:
```python
class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)

model = linear_regression(2, 1)         # 2 đặc trưng vào, 1 giá trị ra
model(X)                                # (3, 1)
```
Tổng quát, `model(X)` với `X` shape `(N, d)` luôn ra `(N, out_features)`.

## 3. Huấn luyện nhiều đầu vào
Không có gì mới về cấu trúc:
```python
dataset = Data2D()                                     # x: (20, 2), y: (20, 1)
model = linear_regression(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
loader = DataLoader(dataset, batch_size=2)

for epoch in range(100):
    for x, y in loader:                  # x: (2, 2), y: (2, 1)
        loss = criterion(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
Với 2 đặc trưng, mô hình là một **mặt phẳng** trong không gian 3 chiều; huấn luyện xoay và dịch mặt phẳng này cho khớp các điểm dữ liệu. Loss giảm dần qua các epoch là dấu hiệu đang hội tụ.

### Bẫy: các đặc trưng trùng nhau
Dữ liệu trong lab có hai cột `x₁` và `x₂` **giống hệt nhau** (cùng là `arange(-1, 1, 0.1)`), và sinh `y = x₁ + x₂ + 1 + nhiễu`. Mô hình học ra `w ≈ [1.40, 0.66]` chứ không phải `[1, 1]`, nhưng **tổng** `w₁ + w₂ ≈ 2.05` và `b ≈ 1.01` đúng như thật. Lý do: khi hai cột bằng nhau, chỉ có tổng `w₁ + w₂` ảnh hưởng đến dự đoán, nên có vô số cặp `(w₁, w₂)` cho kết quả như nhau và gradient descent dừng ở một cặp tùy khởi tạo. Đây là hiện tượng đa cộng tuyến (collinearity): mô hình dự đoán tốt nhưng từng trọng số không đọc ra ý nghĩa được. Hãy nhớ khi diễn giải trọng số của mô hình thật.

## 4. Nhiều đầu ra
Đôi khi cần dự đoán **nhiều** giá trị từ cùng các đặc trưng (ví dụ xe tự lái dự đoán đồng thời góc lái và tốc độ). Mỗi đầu ra có bộ trọng số và bias riêng, dùng chung đầu vào.

Với `d` đặc trưng và `m` đầu ra:
> **Ŷ = X·W + b**, với `W` shape `(d, m)`, `b` shape `(m,)`, `Ŷ` shape `(N, m)`.

Mỗi **cột** của `W` là bộ trọng số của một đầu ra. Một phép nhân ma trận tính tất cả đầu ra một lúc, nhanh hơn viết `m` phương trình riêng.

```python
model = linear_regression(1, 10)                        # 1 đặc trưng vào, 10 đầu ra
model(torch.tensor([[1.0]])).shape                      # (1, 10)
model(torch.tensor([[1.0], [1.0], [3.0]])).shape        # (3, 10): mỗi hàng một mẫu, mỗi cột một đầu ra
model.linear.weight.shape, model.linear.bias.shape      # (10, 1), (10,)   ← (out, in) và (out,)
```

### Huấn luyện nhiều đầu ra
Cùng khung cũ; khác ở dữ liệu và mô hình:
- Nhãn `y` là **vector** cho mỗi mẫu: shape `(N, m)`; dự đoán `ŷ` cũng `(N, m)`.
- `nn.MSELoss()` tính trung bình bình phương sai số trên **mọi** phần tử (mọi mẫu, mọi đầu ra), nên cost gộp sai số của tất cả đầu ra. Giảm cost là cải thiện mọi đầu ra cùng lúc.
- Gradient descent cập nhật cả ma trận trọng số và vector bias bằng phép toán vector hóa.

```python
model = linear_regression(2, 2)                         # 2 đặc trưng, 2 đầu ra
optimizer = optim.SGD(model.parameters(), lr=0.1)
loader = DataLoader(dataset, batch_size=5)              # y có shape (N, 2)
# vòng lặp huấn luyện y hệt trước
```
Trong lab, dữ liệu sinh theo `W = [[1, -1], [1, 3]]`, `b = [1, -1]`. Vì hai cột đặc trưng lại giống nhau, mô hình tìm ra đúng **tổng** trọng số theo từng đầu ra (≈ 2 cho cả hai) và bias `[1, -1]` với loss rất nhỏ.

## 5. `nn.Linear` là viên gạch của mạng nơ-ron
Mạng nơ-ron được xây bằng cách xếp chồng nhiều phép biến đổi tuyến tính xen kẽ với hàm kích hoạt. Hiểu `nn.Linear` nhiều đầu vào, nhiều đầu ra nghĩa là bạn đã hiểu phần "tuyến tính" của mọi lớp fully connected. Ở module cuối của khóa, bạn sẽ thêm hàm kích hoạt sigmoid để làm bài toán phân loại.

## 6. Một vài xu hướng quanh PyTorch (đọc để biết)
Video cuối module điểm qua cách PyTorch được dùng ngày nay; chưa cần làm, chỉ cần biết tên và ý chính:
- **Transfer learning:** thay vì huấn luyện từ đầu, bắt đầu từ mô hình đã học sẵn trên dữ liệu lớn (ví dụ ImageNet cho ảnh, kho văn bản lớn cho ngôn ngữ). Hai cách dùng: **frozen backbone** (giữ cố định các lớp đã học, chỉ huấn luyện lớp đầu ra cho tác vụ mới: nhanh, ít overfitting khi dữ liệu ít) và **fine-tuning** (huấn luyện lớp mới trước, sau đó mở khóa một phần backbone và huấn luyện tiếp với learning rate nhỏ hơn).
- **Hugging Face:** kho hàng nghìn mô hình transformer huấn luyện sẵn, kèm tokenizer và `Trainer` API giúp huấn luyện với rất ít code. Các khóa GenAI sau trong chứng chỉ này dùng nhiều.
- **PyTorch Lightning:** tổ chức mã thành các thành phần (mô hình, bước huấn luyện/validation, logging, checkpoint) để bớt viết vòng lặp tay và dễ mở rộng ra nhiều GPU.
- **Triển khai (deployment):** suy luận (inference) cần tắt các lớp chỉ dùng khi huấn luyện (ví dụ dropout), tối ưu tốc độ và bộ nhớ, gộp đầu vào thành lô. Xuất mô hình qua **TorchScript** hoặc **ONNX** để chạy trên web, di động, cloud mà không cần mã Python huấn luyện.
- Ở quy mô lớn: huấn luyện phân tán nhiều GPU, mixed precision, checkpoint, tích hợp cloud.

## Tóm tắt
- `ŷ = w·x + b`; `X` shape `(N, d)`, mỗi hàng một mẫu; `nn.Linear(d, m)` cho ra `(N, m)`.
- `nn.Linear.weight` có shape `(out_features, in_features)`; `bias` shape `(out_features,)`.
- Huấn luyện nhiều đầu vào/đầu ra dùng đúng khung mini-batch cũ; `MSELoss` gộp sai số mọi đầu ra.
- Đặc trưng trùng nhau làm trọng số không xác định riêng lẻ, chỉ tổng của chúng được xác định.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
