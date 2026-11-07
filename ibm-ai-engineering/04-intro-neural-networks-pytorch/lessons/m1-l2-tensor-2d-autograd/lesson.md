# Module 1 · Bài 2: Tensor 2D và đạo hàm tự động (autograd)

> Nguồn: Lesson 2 của module "Exploring Tensors" (3 video, 2 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~30 phút. Làm lab: ~1 giờ 15 phút. Lab: [`lab-2d.ipynb`](lab-2d.ipynb) và [`lab-autograd.ipynb`](lab-autograd.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Tạo tensor 2D, đọc `shape`, `ndimension`, `numel` và dùng index/slice theo hàng, cột, ma trận con.
2. Thực hiện cộng, nhân vô hướng, Hadamard và **nhân ma trận**; phân biệt `*` với `@`.
3. Hiểu shape và rank quyết định phép toán nào hợp lệ; mở rộng lên tensor 3D (ảnh màu).
4. Giải thích vai trò của đạo hàm trong huấn luyện và dùng `requires_grad`, `backward()`, `.grad`.
5. Hiểu đồ thị tính toán (computational graph), đạo hàm riêng và quy tắc chuỗi (chain rule) mà autograd áp dụng.

## Phần A. Tensor 2D

### 1. Tensor 2D là gì và dùng ở đâu
Tensor 2D là một lưới số có hàng và cột, tức là **matrix**. Trong học máy có hai kiểu dùng chủ đạo:
- **Bảng dữ liệu:** mỗi **hàng** là một mẫu (sample), mỗi **cột** là một đặc trưng (feature).
- **Ảnh xám:** mỗi ô là độ sáng của một pixel.

```python
import torch
A = torch.tensor([[11, 12, 13],
                  [21, 22, 23],
                  [31, 32, 33]])    # list lồng nhau: mỗi list con là một hàng
A.ndimension()   # 2
A.shape          # torch.Size([3, 3])
A.size()         # giống shape
A.numel()        # 9
```
`ndimension()` cho biết số chiều (còn gọi là **rank**; tensor 2D có rank 2). `shape` là kích thước từng chiều.

### 2. Index và slice
Một phần tử cần hai chỉ số: **[hàng, cột]**, đếm từ 0.
```python
A[1, 2]        # hàng 2, cột 3 -> 23
A[1][2]        # cách viết thứ hai, cùng kết quả
A[1, :]        # cả hàng thứ hai
A[:, 2]        # cả cột thứ ba
A[:, 0:2]      # mọi hàng, hai cột đầu (ma trận con)
A[1:3, 2]      # hàng 2 và 3, chỉ cột thứ ba -> tensor([23, 33])
```
**Bẫy:** `A[1:3][2]` khác `A[1:3, 2]`. Vế đầu cắt hàng trước rồi mới index tiếp trên kết quả (lúc này chỉ còn 2 hàng, nên index 2 sai hoặc lấy nhầm hàng). Muốn chọn đồng thời hàng và cột, hãy đặt cả hai trong **một cặp ngoặc** cách nhau bằng dấu phẩy.

### 3. Các phép toán
Với `X`, `Y` cùng kích thước:
```python
X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])
X + Y      # cộng từng phần tử, cần cùng kích thước (hoặc broadcasting)
2 * Y      # nhân vô hướng: mọi phần tử nhân 2
X * Y      # Hadamard: nhân từng phần tử cùng vị trí. KHÔNG phải nhân ma trận
```
**Nhân ma trận** theo quy tắc đại số tuyến tính: ô (i, j) của kết quả là tích vô hướng giữa hàng i của ma trận trái và cột j của ma trận phải.
```python
A = torch.tensor([[0, 1, 1], [1, 0, 1]])     # (2, 3)
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])  # (3, 2)
A @ B          # hoặc torch.mm(A, B)  -> kết quả (2, 2)
```
Quy tắc kích thước: **(m, n) @ (n, p) → (m, p)**. Số cột của ma trận trái phải bằng số hàng của ma trận phải, nếu không PyTorch báo lỗi. Thứ tự có ý nghĩa: `A @ B` hợp lệ không có nghĩa là `B @ A` hợp lệ, và kết quả nói chung khác nhau.

Nhân ma trận chính là phép tính trong lớp **fully connected**: đầu vào (nhiều mẫu × nhiều đặc trưng) nhân ma trận trọng số rồi cộng bias. Vì vậy hãy luôn kiểm tra `shape` trước khi nhân.

### 4. Lên 3D và hơn nữa
Thêm một chiều ta được tensor 3D. Ví dụ ảnh màu RGB là tensor `(cao, rộng, 3)`: kênh 0 là đỏ, kênh 1 là lục, kênh 2 là lam. Cộng, nhân vô hướng mở rộng tự nhiên sang 3D miễn là kích thước khớp. Tensor nhiều chiều là nền của CNN, xử lý ảnh và video.

Lưu ý hay gặp ở bước sau: thư viện ảnh thường để kênh ở cuối `(H, W, C)`, còn PyTorch thường để kênh ở đầu `(C, H, W)`. Đổi thứ tự trục bằng `permute`.

### 5. Chuyển từ Pandas
Giống 1D: lấy `df.values` (NumPy array 2D) rồi `torch.from_numpy`. Hàng của DataFrame thành hàng của tensor. Kiểu dữ liệu đi theo NumPy, nên cần `.float()` nếu muốn tính gradient.

## Phần B. Đạo hàm tự động (autograd)

### 6. Vì sao cần đạo hàm?
Huấn luyện mô hình là điều chỉnh tham số để **loss** (độ sai của dự đoán) nhỏ đi. Muốn biết nên chỉnh tham số nào, theo hướng nào, ta cần biết loss thay đổi bao nhiêu khi tham số đổi một chút. Đó chính là **đạo hàm**. Với nhiều tham số, tập hợp các đạo hàm riêng gọi là **gradient**.

Mạng nơ-ron có thể có hàng triệu tham số và nhiều lớp chồng nhau. Tự tính đạo hàm bằng tay là bất khả thi, nên PyTorch có hệ thống **autograd** làm việc này tự động.

### 7. Cách autograd hoạt động
1. Tạo tensor với `requires_grad=True`. PyTorch sẽ ghi lại mọi phép toán dùng đến nó.
2. Các phép toán đó tạo thành một **đồ thị tính toán** (computational graph): mỗi nút là một phép toán, các cạnh cho biết giá trị chảy từ đâu đến đâu.
3. Gọi `.backward()` trên kết quả (thường là loss). PyTorch đi ngược đồ thị và tính đạo hàm.
4. Kết quả nằm ở thuộc tính `.grad` của các tensor có `requires_grad=True`.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 2 * x + 1       # y = x² + 2x + 1
y.backward()
x.grad                        # tensor(6.), vì dy/dx = 2x + 2 = 6 tại x = 2
```

Thực tế, tham số của mô hình (weight, bias) là những tensor có `requires_grad=True`, còn dữ liệu đầu vào thì thường không cần.

### 8. Đạo hàm riêng
Khi hàm có nhiều biến, đạo hàm riêng đo sự thay đổi theo **một** biến trong khi giữ các biến khác cố định.
```python
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u * v + u ** 2            # f(u, v) = uv + u²
f.backward()
u.grad    # 4.  (df/du = v + 2u = 2 + 2)
v.grad    # 1.  (df/dv = u)
```

### 9. Quy tắc chuỗi (chain rule)
Mạng có nhiều lớp chồng nhau, tức là hàm lồng hàm: `y = g(f(x))`. Quy tắc chuỗi nói đạo hàm bằng tích các đạo hàm thành phần: `dy/dx = g'(f(x)) · f'(x)`. Autograd tự áp dụng quy tắc này khi đi ngược đồ thị, nên mạng sâu bao nhiêu lớp cũng tính được.

### 10. Vài điều cần nhớ khi dùng autograd
- `backward()` chỉ dùng trực tiếp cho kết quả là **một số (scalar)**. Với kết quả là vector, gộp thành scalar trước, ví dụ `y.sum()`.
- Gradient bị **cộng dồn**: gọi `backward()` nhiều lần thì `.grad` cộng thêm vào giá trị cũ. Trong vòng lặp huấn luyện phải đặt lại về 0 (`x.grad.zero_()`) trước mỗi lượt.
- Muốn chuyển tensor đang theo dõi gradient sang NumPy (để vẽ đồ thị), dùng `.detach().numpy()`.
- Chỉ tensor "lá" (tạo trực tiếp, không phải kết quả của phép toán) mới tự giữ `.grad`. Nút trung gian thì không, trừ khi gọi `retain_grad()`.
- Hàm ReLU `max(0, x)` có đạo hàm 0 với x âm và 1 với x dương. Đây là hàm kích hoạt rất phổ biến, ta sẽ gặp lại.

## Tóm tắt
- Tensor 2D = matrix: hàng là mẫu, cột là đặc trưng. Index theo `[hàng, cột]`.
- `*` là Hadamard, `@` là nhân ma trận, quy tắc `(m, n) @ (n, p) → (m, p)`.
- Autograd ghi lại phép toán thành đồ thị, rồi `backward()` đi ngược để tính gradient bằng chain rule. Kết quả nằm trong `.grad`, và gradient cộng dồn nên phải đặt lại về 0.
- Bài sau (module 2) dùng những công cụ này để xây dataset; từ module 3 ta dùng gradient để huấn luyện mô hình thật.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md) (đáp án ở [`quiz-answers.md`](quiz-answers.md)), rồi làm hai lab. Lời giải ở [`solution/`](solution/), chỉ mở khi bạn đã thử xong hoặc dùng hết gợi ý.
