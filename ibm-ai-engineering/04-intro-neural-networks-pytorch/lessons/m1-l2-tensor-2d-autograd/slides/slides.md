---
marp: true
theme: default
paginate: true
title: Tensor 2D và autograd
---

# Tensor 2D và đạo hàm tự động
Module 1 · Bài 2 · Introduction to Neural Networks and PyTorch

---

# Tensor 2D là một bảng
```python
A = torch.tensor([[11, 12, 13],
                  [21, 22, 23],
                  [31, 32, 33]])
A.ndimension()  # 2
A.shape         # torch.Size([3, 3])
A.numel()       # 9
```
Hàng = mẫu · Cột = đặc trưng · Ảnh xám = lưới pixel

---

# Index và slice: [hàng, cột]
```python
A[1, 2]      # 23
A[1, :]      # cả hàng thứ hai
A[:, 2]      # cả cột thứ ba
A[:, 0:2]    # ma trận con
A[1:3, 2]    # tensor([23, 33])
```
⚠️ `A[1:3][2]` ≠ `A[1:3, 2]`: cắt hàng trước rồi mới index tiếp

---

# Cộng, nhân vô hướng, Hadamard
```python
X + Y     # từng phần tử, cùng kích thước
2 * Y     # nhân vô hướng
X * Y     # Hadamard, KHÔNG phải nhân ma trận
```

---

# Nhân ma trận
**(m, n) @ (n, p) → (m, p)**
```python
A = torch.tensor([[0, 1, 1], [1, 0, 1]])        # (2, 3)
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])     # (3, 2)
A @ B                                           # (2, 2)
```
- Ô (i, j) = dot(hàng i của A, cột j của B)
- Thứ tự quan trọng: `A @ B` ≠ `B @ A`
- Chính là phép tính trong lớp fully connected

---

# Lên 3D
Ảnh màu RGB = `(cao, rộng, 3)`: kênh 0 đỏ · 1 lục · 2 lam

| Thư viện ảnh | PyTorch |
|---|---|
| `(H, W, C)` | `(C, H, W)` |

Đổi trục: `img.permute(2, 0, 1)`

---

# Vì sao cần đạo hàm?
Huấn luyện = chỉnh tham số để **loss** nhỏ đi.

Đạo hàm cho biết: *tham số đổi một chút thì loss đổi bao nhiêu*.

Hàng triệu tham số → máy phải tự tính → **autograd**

---

# Autograd: 4 bước
1. `x = torch.tensor(2.0, requires_grad=True)`
2. Tính `y = x**2 + 2*x + 1` (PyTorch ghi lại đồ thị)
3. `y.backward()`
4. Đọc `x.grad` → `tensor(6.)`

Kiểm tra tay: dy/dx = 2x + 2 = 6 ✅

---

# Đồ thị tính toán
```
        ┌── x² ──┐
   x ───┤         ├── (+) ── (+1) ── y
        └── 2x ───┘
```
(đồ thị của y = x² + 2x + 1)

Mỗi nút là một phép toán. `backward()` đi **ngược** từ y về x.

---

# Đạo hàm riêng
f(u, v) = u·v + u²  tại u = 1, v = 2

```python
f.backward()
u.grad   # 4  (= v + 2u)
v.grad   # 1  (= u)
```
Một lần `backward`, ra gradient của mọi tham số.

---

# Chain rule
y = g(f(x))  →  dy/dx = g′(f(x)) · f′(x)

Mạng nhiều lớp = hàm lồng hàm.
Autograd tự áp dụng chain rule khi đi ngược đồ thị.

---

# Ba điều cần nhớ
1. `backward()` cần kết quả là **scalar** (vector → `.sum()`)
2. Gradient **cộng dồn** → đặt lại: `x.grad.zero_()`
3. Vẽ đồ thị: `x.detach().numpy()`

---

# Tóm tắt
- Tensor 2D: hàng = mẫu, cột = đặc trưng
- `*` Hadamard · `@` nhân ma trận · `(m,n)@(n,p)→(m,p)`
- Autograd: ghi đồ thị → `backward()` → đọc `.grad`
- Tiếp theo: hai lab, rồi module 2 (Dataset)
