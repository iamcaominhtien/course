# Module 1 · Bài 1: Tensor 1D trong PyTorch

> Nguồn: Lesson 1 của module "Exploring Tensors" (5 video, 1 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~25 phút. Làm lab: ~45 phút. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Nói được tensor là gì và vì sao mạng nơ-ron "nói bằng tensor".
2. Tạo tensor 1D, kiểm tra `dtype`, kích thước, đổi kiểu dữ liệu và đổi hình dạng (`view`).
3. Truy cập và sửa phần tử bằng index và slice.
4. Chuyển qua lại giữa tensor, NumPy và Pandas, và biết khi nào chúng dùng chung bộ nhớ.
5. Dùng các phép toán 1D: cộng, nhân vô hướng, nhân từng phần tử (Hadamard), tích vô hướng (dot), broadcasting, hàm toán học và `linspace`.

## 1. Vì sao bắt đầu từ tensor?
Máy tính không "thấy" con mèo trong ảnh. Nó chỉ thấy một dãy số. Mạng nơ-ron nhận số vào, nhân với các tham số, rồi trả số ra. Mọi thứ trong quá trình đó (dữ liệu đầu vào, tham số, kết quả trung gian, dự đoán) đều nằm trong **tensor**.

Bạn không cần nhớ các kiến trúc hiện đại ngay, chỉ cần biết bức tranh: mạng feedforward sâu là nền tảng, CNN tận dụng cấu trúc không gian của ảnh, RNN/LSTM/GRU xử lý chuỗi theo thứ tự, còn Transformer dùng attention để nhìn cả chuỗi cùng lúc. Kiến trúc nào cũng chạy trên tensor.

## 2. Scalar, vector, matrix, tensor
| Tên | Số chiều | Ví dụ |
|---|---|---|
| Scalar | 0 | learning rate `0.01`, nhiệt độ |
| Vector | 1 | một hàng đặc trưng (feature) của một mẫu: `[1.2, 0.7, 3.1]` |
| Matrix | 2 | bảng dữ liệu: mỗi hàng một mẫu, mỗi cột một đặc trưng |
| Tensor | n | ảnh màu (cao × rộng × kênh), lô nhiều ảnh... |

**Tensor** là cách gọi chung cho mảng nhiều chiều. Scalar, vector, matrix chỉ là các trường hợp riêng của nó.

Hai ý hình học đáng nhớ:
- Vector là một mũi tên có độ dài và hướng. Matrix là "cỗ máy" biến một vector thành vector khác (xoay, co giãn, phản chiếu, làm nghiêng). Đó là cách lớp nơ-ron biến đổi dữ liệu.
- Tích vô hướng (dot) đo mức độ "cùng hướng" của hai vector: dương là cùng hướng, bằng 0 là vuông góc, âm là ngược hướng.

## 3. Tensor khác mảng NumPy ở đâu?
Về cách dùng, tensor rất giống `ndarray`. Hai điểm khác làm nên PyTorch:
- **Autograd:** PyTorch có thể tự tính đạo hàm (sẽ học ở bài 2).
- **Thiết bị:** tensor có thể chạy trên CPU hoặc GPU.

Trong một mạng nơ-ron, tensor xuất hiện ở mọi bước: dữ liệu vào, tham số (weight, bias), kết quả trung gian và đầu ra. Dữ liệu thật thường phải đổi sang tensor trước khi đưa vào mô hình, và kích thước phải khớp với lớp nhận nó. Với dữ liệu lớn, ta không nạp hết vào bộ nhớ mà dùng lớp `Dataset` để lấy từng mẫu khi cần (học kỹ ở module 2).

## 4. Tensor 1D
Tensor 1D là một dãy số xếp theo một chiều, tương đương vector hay mảng 1 chiều. Mọi phần tử cùng một kiểu dữ liệu.

### 4.1 Tạo tensor
```python
import torch

a = torch.tensor([0, 1, 2, 3, 4])        # từ list số nguyên
b = torch.tensor([0.0, 1.0, 2.0])        # từ list số thực
z = torch.zeros(5)                        # toàn số 0
o = torch.ones(5)                         # toàn số 1
r = torch.rand(5)                         # ngẫu nhiên trong [0, 1)
```

### 4.2 Kiểu dữ liệu (`dtype`) và kích thước
```python
a.dtype        # torch.int64  (list số nguyên -> int64)
b.dtype        # torch.float32 (list số thực -> float32)
a.size()       # torch.Size([5])
a.ndimension() # 1
a.numel()      # 5, tổng số phần tử
```
- Tensor số thực (`float32`) mới tính được gradient, nên là kiểu chuẩn khi huấn luyện.
- Tensor số nguyên hay dùng làm chỉ số (index) hoặc nhãn lớp.
- Đổi kiểu bằng `.float()`, `.long()` hoặc `.to(torch.float32)`. Các lệnh này tạo tensor mới, không sửa tensor cũ.
- Chọn sai `dtype` gây lỗi hoặc kết quả lạ. Ví dụ `torch.tensor([2, 4, 6]).mean()` báo lỗi vì trung bình cần kiểu số thực.

### 4.3 Index và slice
Index bắt đầu từ 0, giống Python.
```python
t = torch.tensor([20, 1, 2, 3, 4])
t[0]          # tensor(20)
t[1:4]        # tensor([1, 2, 3]), lấy index 1, 2, 3 (cận phải không gồm)
t[0] = 100    # sửa một phần tử
t[3:5] = torch.tensor([300, 400])      # sửa cả một đoạn
t[[1, 3]] = 0                          # sửa nhiều vị trí bằng list index
```
- Slice là một **view** (nhìn vào cùng dữ liệu), không sao chép. Sửa slice thì tensor gốc đổi theo.
- Truy cập ngoài phạm vi (`t[5]` với tensor có 5 phần tử) báo lỗi.
- Lấy giá trị Python thuần từ tensor một phần tử bằng `.item()`; chuyển cả tensor thành list bằng `.tolist()`.

### 4.4 Đổi hình dạng với `view`
Nhiều lớp mạng cần đầu vào 2D. `view` đổi cách đọc dữ liệu mà không sao chép, miễn là tổng số phần tử không đổi.
```python
v = torch.tensor([0., 1., 2., 3., 4.])   # shape (5,)
v.view(5, 1)      # cột, shape (5, 1)
v.view(1, 5)      # hàng, shape (1, 5)
v.view(-1, 1)     # -1 = "tự suy ra", kết quả (5, 1)
```
Chỉ được dùng `-1` ở **một** tham số.

### 4.5 NumPy và Pandas
```python
import numpy as np, pandas as pd

arr = np.array([0.0, 1.0, 2.0])
t = torch.from_numpy(arr)       # tensor dùng CHUNG bộ nhớ với arr
back = t.numpy()                # NumPy array dùng chung bộ nhớ với t
s = pd.Series([0.1, 2, 0.3])
ts = torch.from_numpy(s.values) # Series -> tensor qua .values
```
**Bẫy quan trọng:** `from_numpy` và `.numpy()` không sao chép. Sửa `arr` thì `t` đổi theo, và ngược lại. Muốn bản sao độc lập, dùng `torch.tensor(arr)` hoặc `.clone()`.

## 5. Phép toán trên tensor 1D
Cho `u = [1, 2, 3]` và `v = [4, 5, 6]`:

| Phép toán | Cú pháp | Kết quả | Dùng để |
|---|---|---|---|
| Cộng (từng phần tử) | `u + v` | `[5, 7, 9]` | gộp đặc trưng, cập nhật tham số |
| Nhân vô hướng | `2 * u` | `[2, 4, 6]` | chuẩn hóa, nhân learning rate với gradient |
| Hadamard (nhân từng phần tử) | `u * v` | `[4, 10, 18]` | gán trọng số, mask |
| Tích vô hướng (dot) | `torch.dot(u, v)` | `32` (một số) | mô hình tuyến tính, đo độ giống nhau |

Điểm cần phân biệt: `u * v` cho ra tensor, còn `torch.dot(u, v)` cho ra **một số**. `torch.dot` yêu cầu hai tensor cùng `dtype` và cùng độ dài.

### Broadcasting
Khi kích thước khác nhau, PyTorch có thể "kéo giãn" tensor nhỏ cho khớp với tensor lớn theo quy tắc cố định. Ví dụ phổ biến nhất: cộng một scalar vào mọi phần tử (`u + 10`), hoặc cộng bias vào cả một lô kết quả. Tiện, nhưng nếu kích thước không như bạn tưởng, kết quả vẫn có thể "đúng cú pháp mà sai nghĩa". Luôn kiểm tra `shape` trước.

### Hàm toán học và `linspace`
```python
x = torch.tensor([1.0, -1.0, 1.0, -1.0])
x.mean()   x.std()   x.max()   x.min()     # rút gọn về một số
torch.sin(torch.tensor([0, np.pi/2, np.pi]))  # áp dụng lên từng phần tử
torch.linspace(0, 2*np.pi, 100)               # 100 điểm cách đều từ 0 đến 2π
```
- `mean`, `std`, `max`, `min` gộp cả tensor thành một số. `sin`, `cos`, `exp` áp dụng lên **từng phần tử**.
- `max()` trả về giá trị lớn nhất, không phải vị trí. Muốn vị trí dùng `argmax()`.
- `linspace` kết hợp với `sin` là cách nhanh để vẽ đồ thị một hàm. Nhớ đổi sang NumPy (`.numpy()`) trước khi đưa vào `matplotlib`.

## 6. Tóm tắt
- Tensor là cấu trúc dữ liệu trung tâm của PyTorch: dữ liệu, tham số, kết quả trung gian và đầu ra đều là tensor.
- Tensor 1D = vector. Luôn kiểm tra `dtype` và `shape` trước khi tính toán.
- `view`/`reshape` đổi hình dạng, `from_numpy`/`.numpy()` dùng chung bộ nhớ, slice là view.
- Cộng, nhân vô hướng, Hadamard, dot, broadcasting, hàm toán học và `linspace` là bộ công cụ nền cho mọi thứ phía sau.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md) (đáp án ở [`quiz-answers.md`](quiz-answers.md), đừng xem trước), rồi làm [`lab.ipynb`](lab.ipynb). Lời giải nằm ở [`solution/`](solution/), chỉ mở khi bạn đã thử xong hoặc đã dùng hết gợi ý.
