---
marp: true
theme: default
paginate: true
title: Tensor 1D trong PyTorch
---

# Tensor 1D trong PyTorch
Module 1 · Bài 1 · Introduction to Neural Networks and PyTorch

---

# Máy tính chỉ thấy số
- Ảnh con mèo → một dãy số
- Mạng nơ-ron: số vào → nhân tham số → số ra
- Dữ liệu, tham số, kết quả trung gian, dự đoán: **đều là tensor**

---

# Bức tranh các kiến trúc
| Kiến trúc | Dành cho | Ý chính |
|---|---|---|
| Feedforward sâu | dữ liệu bảng, nền tảng | nhiều lớp ẩn, đi một chiều |
| CNN | ảnh | bộ lọc nhỏ, dùng chung tham số |
| RNN / LSTM / GRU | chuỗi | trạng thái ẩn, nhớ thứ tự |
| Transformer | chuỗi dài, ngôn ngữ | attention, tính song song |

Kiến trúc nào cũng chạy trên tensor.

---

# Scalar → vector → matrix → tensor
| | Chiều | Ví dụ |
|---|---|---|
| Scalar | 0 | learning rate |
| Vector | 1 | một mẫu: các đặc trưng |
| Matrix | 2 | bảng: mẫu × đặc trưng |
| Tensor | n | ảnh màu: cao × rộng × kênh |

---

# Tạo tensor 1D
```python
a = torch.tensor([0, 1, 2, 3, 4])      # int64
b = torch.tensor([0.0, 1.0, 2.0])      # float32
torch.zeros(5);  torch.ones(5);  torch.rand(5)
```
- list số nguyên → `int64`, list số thực → `float32`
- Huấn luyện cần **float** (mới có gradient)

---

# dtype và kích thước
```python
a.dtype        # torch.int64
a.size()       # torch.Size([5])
a.numel()      # 5
a.float()      # tensor MỚI, kiểu float32
```
⚠️ `torch.tensor([2, 4, 6]).mean()` → lỗi. Đổi sang float trước.

---

# Index, slice, view
```python
t[0]        t[1:4]        t[[1, 3]] = 0
v.view(-1, 1)        # vector → cột, (5,) → (5, 1)
```
- Index từ 0; slice `1:4` lấy index 1, 2, 3
- Slice là **view**: sửa slice, gốc đổi theo
- `view` giữ nguyên số phần tử; `-1` chỉ dùng một lần

---

# NumPy và Pandas
```python
t = torch.from_numpy(arr)    # CHUNG bộ nhớ
arr[:] = 0                   # t cũng thành 0!
t2 = torch.tensor(arr)       # bản sao độc lập
torch.from_numpy(series.values)
```
⚠️ Muốn tách riêng: `torch.tensor(...)` hoặc `.clone()`

---

# Bốn phép toán nền tảng
`u = [1, 2, 3]`, `v = [4, 5, 6]`

| Phép toán | Kết quả |
|---|---|
| `u + v` | `[5, 7, 9]` |
| `2 * u` | `[2, 4, 6]` |
| `u * v` (Hadamard) | `[4, 10, 18]` |
| `torch.dot(u, v)` | `32` (một số) |

---

# Broadcasting, hàm toán học, linspace
```python
u + 10                                  # broadcasting
torch.sin(x)                            # từng phần tử
x.mean()  x.std()  x.max()  x.min()     # gộp về một số
torch.linspace(0, 2*np.pi, 100)         # 100 điểm cách đều
plt.plot(x.numpy(), torch.sin(x).numpy())
```

---

# Tóm tắt
- Tensor là cấu trúc dữ liệu trung tâm của PyTorch
- Luôn kiểm tra `dtype` và `shape`
- Slice và `from_numpy` **dùng chung bộ nhớ**
- Cộng · nhân vô hướng · Hadamard · dot
- Tiếp theo: tensor 2D và autograd → làm lab!
