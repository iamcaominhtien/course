---
marp: true
theme: default
paginate: true
title: Huấn luyện theo kiểu PyTorch
---

# Huấn luyện theo kiểu PyTorch
Module 4 · Introduction to Neural Networks and PyTorch

---

# SGD: một mẫu mỗi lần cập nhật
dự đoán → loss → gradient → cập nhật → mẫu kế tiếp
- 1 epoch = 1 lượt qua hết dữ liệu
- Cập nhật nhiều, tiến nhanh
- Loss **dao động**, xu hướng vẫn đi xuống

| 10 epoch, lr = 0.1 | Cập nhật | Cost cuối |
|---|---|---|
| Batch | 10 | ≈ 1.6 |
| SGD | 600 | ≈ 0.013 |

---

# Mini-batch
Batch size · Iteration · Epoch

iteration mỗi epoch = số mẫu / batch size

| 60 mẫu | iteration / epoch |
|---|---|
| batch 5 | 12 |
| batch 10 | 6 |
| batch 20 | 3 |

---

# Batch nhỏ vs lớn
| Nhỏ | Lớn |
|---|---|
| cập nhật nhiều | cập nhật ít |
| nhiễu | mượt, ổn định |
| ít bộ nhớ | nhiều bộ nhớ |

Mini-batch: phổ biến nhất

---

# `DataLoader`
```python
loader = DataLoader(dataset, batch_size=5, shuffle=True)
for x, y in loader:     # x: (5, 1), y: (5, 1)
    ...
len(loader)             # số iteration / epoch
```
- `batch_size=1` → SGD · `batch_size=len(ds)` → batch GD
- `drop_last=True` bỏ lô cuối nhỏ hơn

---

# Optimizer
```python
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    for x, y in trainloader:
        loss = criterion(model(x), y)   # forward + loss
        optimizer.zero_grad()           # xóa gradient cũ
        loss.backward()                 # tính gradient
        optimizer.step()                # cập nhật
```

---

# Train / Validation / Test
| Tập | Dùng để |
|---|---|
| Train | học tham số |
| Validation | chọn siêu tham số, chọn mô hình |
| Test | đo cuối cùng, **một lần** |

---

# Overfitting
Train loss ↓ nhưng **validation loss ↑**

→ mô hình học thuộc nhiễu, dự đoán dữ liệu mới kém

Chọn mô hình theo **validation loss**

---

# Chọn learning rate
`lr ∈ {0.0001, 0.001, 0.01, 0.1}` → train xong đo validation

Mô hình tốt nhất trên **train** ≠ tốt nhất trên **validation**

```python
with torch.no_grad():          # đánh giá: không cập nhật
    val_loss = criterion(model(val.x), val.y)
```

---

# Lưu và nạp mô hình
```python
torch.save(model.state_dict(), "best.pt")

m = linear_regression(1, 1)           # cùng kiến trúc
m.load_state_dict(torch.load("best.pt"))
m.eval()
```

---

# Tóm tắt
- Mini-batch + `DataLoader`
- Vòng lặp: forward → loss → `zero_grad` → `backward` → `step`
- Train / validation / test; chọn theo validation
- `state_dict` để lưu và nạp
