---
marp: true
theme: default
paginate: true
title: Hồi quy tuyến tính — dự đoán và loss
---

# Hồi quy tuyến tính: dự đoán và loss
Module 3 · Bài 1 · Introduction to Neural Networks and PyTorch

---

# Bài toán
Dữ liệu: `(x₁, y₁), …, (xₙ, yₙ)` → học có giám sát

**ŷ = w · x + b**
- `w` (weight): độ dốc
- `b` (bias): hệ số chặn

Huấn luyện = tìm `w`, `b` khớp dữ liệu nhất

---

# Nhiễu (noise)
- Dữ liệu thật lệch khỏi đường thẳng: sai số đo, yếu tố ẩn
- Giả định Gauss: lệch nhỏ phổ biến, lệch lớn hiếm
- Mục tiêu: bắt **xu hướng chung**, không khớp từng điểm

---

# Dự đoán bằng tensor
```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)
def forward(x): return w * x + b

forward(torch.tensor([[1.0], [2.0]]))   # [[1.], [3.]]
```
Mỗi hàng của `x` là một mẫu → mỗi mẫu một dự đoán

---

# `nn.Linear`
```python
torch.manual_seed(1)
model = Linear(in_features=1, out_features=1)
list(model.parameters())      # w ≈ 0.5153, b ≈ −0.4414
model(torch.tensor([[1.0], [2.0]]))
model.state_dict()            # weight, bias
```

---

# Mô hình tự viết: `nn.Module`
```python
class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)
```
Gọi `model(x)`, không gọi `model.forward(x)`

---

# Hàm loss: MSE
**MSE = (1/N) · Σ (ŷᵢ − yᵢ)²**
- Bình phương: không triệt tiêu, phạt nặng sai số lớn
```python
nn.MSELoss()(torch.tensor([1., 2., 3.]), torch.tensor([1., 1., 5.]))
# (0 + 1 + 4) / 3 = 1.6667
```

---

# Loss là hàm của tham số
Đổi `w`, `b` → dự đoán đổi → loss đổi

Huấn luyện = tìm `w`, `b` làm loss **nhỏ nhất**
(bài sau: gradient descent)

---

# Tóm tắt
- `ŷ = w·x + b`, forward pass = dự đoán
- `nn.Linear`, `nn.Module`, `state_dict()`
- MSE đo sai số; huấn luyện = tối ưu loss
