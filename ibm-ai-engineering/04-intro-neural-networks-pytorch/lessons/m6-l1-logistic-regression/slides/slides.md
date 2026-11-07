---
marp: true
theme: default
paginate: true
title: Logistic regression — dự đoán
---

# Logistic regression: dự đoán
Module 6 · Introduction to Neural Networks and PyTorch

---

# Phân loại
- Đầu ra: **nhãn rời rạc** (0, 1, 2…)
- `X`: mỗi hàng một mẫu, mỗi cột một đặc trưng
- `y`: nhãn của từng mẫu
- Điểm số: **z = w · x + b**

---

# Decision boundary
Tại **z = 0**

| Đặc trưng | Ranh giới |
|---|---|
| 1 | điểm |
| 2 | đường thẳng |
| 3 | mặt phẳng |
| nhiều | siêu phẳng |

Dữ liệu tách được bằng một đường: **tách tuyến tính**

---

# Hạn chế của ngưỡng cứng
`z > 0 → lớp 1` · `z ≤ 0 → lớp 0`

Nhận được nhãn, nhưng **không biết độ tự tin**

---

# Sigmoid
**p = 1 / (1 + e^(−z))**

- z rất âm → p ≈ 0
- z = 0 → p = 0.5
- z rất dương → p ≈ 1

p = xác suất thuộc lớp 1

---

# Ngưỡng và độ tự tin
- `p ≥ 0.5` → lớp 1 · `p < 0.5` → lớp 0
- Ngưỡng 0.5 ⇔ `z = 0` ⇔ decision boundary

| p | Ý nghĩa |
|---|---|
| 0.95 | rất chắc lớp 1 |
| 0.52 | sát ranh giới, lưỡng lự |
| 0.03 | rất chắc lớp 0 |

---

# `nn.Sequential`
```python
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
model(torch.tensor([[1.0], [100.0]]))      # (2, 1), xác suất
model.state_dict().keys()                  # '0.weight', '0.bias'
```
2 đặc trưng: `nn.Linear(2, 1)`

---

# Lớp `nn.Module` tự viết
```python
class logistic_regression(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```
Cùng tính toán với `Sequential`; tên tham số: `linear.weight`

---

# Lưu ý
- Chưa huấn luyện → tham số ngẫu nhiên → xác suất "ngẫu nhiên"
- Đầu vào lớn (±100) → sigmoid **bão hòa** (≈ 0 hoặc 1)

---

# Tóm tắt
- Phân loại: nhãn rời rạc, ranh giới tại `z = 0`
- Sigmoid → xác suất; ngưỡng 0.5 → nhãn
- Logistic regression = `Linear` + `Sigmoid`
- Tiếp theo: huấn luyện đúng cách (cross-entropy)
