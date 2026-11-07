---
marp: true
theme: default
paginate: true
title: Hồi quy nhiều đầu vào, nhiều đầu ra
---

# Nhiều đầu vào, nhiều đầu ra
Module 5 · Introduction to Neural Networks and PyTorch

---

# Nhiều đầu vào
**ŷ = w · x + b** (`d` đặc trưng → `d` trọng số + 1 bias)

- Cost: vẫn MSE
- Gradient descent cập nhật cả **vector** `w`

---

# Shape
| | shape |
|---|---|
| một mẫu `x` | `(1, d)` |
| `w` | `(d, 1)` |
| `x @ w` | `(1, 1)` |
| nhiều mẫu `X` | `(N, d)` — mỗi hàng một mẫu |
| `X @ w` | `(N, 1)` |

Số cột của `x` = số hàng của `w`

---

# Dự đoán bằng tay
```python
w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)
forward = lambda x: torch.mm(x, w) + b

forward(torch.tensor([[1., 2.]]))      # [[9.]]
forward(X)   # X = [[1,1],[1,2],[1,3]] → [[6.],[9.],[12.]]
```

---

# `nn.Linear` nhiều đầu vào
```python
model = nn.Linear(in_features=2, out_features=1)
model(X)                 # (3, 2) → (3, 1)
model.weight.shape       # (1, 2)  = (out, in) !
model.bias.shape         # (1,)
```
⚠️ PyTorch lưu `weight` dạng `(out, in)`: chuyển vị so với `W` viết tay

---

# Huấn luyện
```python
model = linear_regression(2, 1)
loader = DataLoader(dataset, batch_size=2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
for epoch in range(100):
    for x, y in loader:
        loss = criterion(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
2 đặc trưng = một **mặt phẳng** trong không gian 3D

---

# Bẫy: đặc trưng trùng nhau
Lab: hai cột `x₁ = x₂`, `y = x₁ + x₂ + 1 + nhiễu`

Học ra `w ≈ [1.40, 0.66]`, **tổng ≈ 2.05**, `b ≈ 1.01`

Chỉ tổng `w₁ + w₂` ảnh hưởng → dự đoán tốt, trọng số riêng lẻ **không đọc được**

---

# Nhiều đầu ra
**Ŷ = X · W + b** · `W: (d, m)`, `b: (m,)`, `Ŷ: (N, m)`

```python
model = linear_regression(1, 10)
model(torch.tensor([[1.0]])).shape                  # (1, 10)
model(torch.tensor([[1.], [1.], [3.]])).shape       # (3, 10)
model.linear.weight.shape                           # (10, 1)
```
Mỗi cột của `W` là một đầu ra

---

# Huấn luyện nhiều đầu ra
- `y` shape `(N, m)`
- `MSELoss`: trung bình trên **mọi** phần tử → cải thiện mọi đầu ra cùng lúc
- Vòng lặp huấn luyện **y hệt** trước

---

# Xu hướng quanh PyTorch
- **Transfer learning:** frozen backbone · fine-tuning
- **Hugging Face:** mô hình transformer huấn luyện sẵn
- **Lightning:** tổ chức mã huấn luyện
- **Triển khai:** `eval`, TorchScript, ONNX

---

# Tóm tắt
- Shape quyết định mọi thứ: `(N, d) → (N, m)`
- `nn.Linear.weight` là `(out, in)`
- Khung huấn luyện không đổi
- Đặc trưng trùng nhau → trọng số không xác định riêng lẻ
