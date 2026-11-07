---
marp: true
theme: default
paginate: true
title: Cross-entropy và huấn luyện logistic regression
---

# Cross-entropy và huấn luyện logistic regression
Module 6 · Introduction to Neural Networks and PyTorch

---

# Thí nghiệm: khởi tạo xấu
Dữ liệu: `x ∈ [−1, 1)`, nhãn 1 khi `x > 0.2` · init `w = −5, b = −10` · `lr = 2`

| Loss | gradient `w` | loss đầu → cuối | accuracy |
|---|---|---|---|
| MSE | ≈ −0.00001 | 0.35 → 0.35 | **65%** |
| Cross-entropy | ≈ −0.21 | 4.55 → 0.05 | **100%** |

Chỉ đổi hàm loss!

---

# Vì sao MSE kẹt?
- Sigmoid **bão hòa** → xác suất sát 0/1
- MSE trên xác suất: vùng **phẳng**, gradient ≈ 0
- Gradient descent không biết đi hướng nào

---

# Bernoulli và likelihood
Hai kết quả `1` / `0`, một tham số θ: `P(1) = θ`

Ngửa, ngửa, sấp với θ = 0.2:
`0.2 × 0.2 × 0.8 = 0.032` ← **likelihood**

---

# MLE và log-likelihood
**MLE:** chọn tham số làm likelihood **lớn nhất**

Dùng **log**: tích → tổng, dễ tính, cùng điểm cực đại

---

# Cross-entropy
**loss = − (1/N) Σ [ y·log p + (1−y)·log(1−p) ]**

- `y = 1`: chỉ `−log p` (muốn p → 1)
- `y = 0`: chỉ `−log(1−p)` (muốn p → 0)

```python
nn.BCELoss()(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([1., 1., 0.]))  # 1.339
```

---

# Phạt nặng khi tự tin mà sai
Nhãn thật `y = 1`:

| dự đoán `p` | loss |
|---|---|
| 0.9 | 0.105 |
| 0.1 | 2.303 (> 20×) |

Trơn · phạt mạnh · có diễn giải xác suất

---

# Huấn luyện trong PyTorch
```python
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# vòng lặp: forward → loss → zero_grad → backward → step
labels = (model(x) >= 0.5).float()
accuracy = (labels == y).float().mean()
```

---

# Optimizer & kỹ thuật nâng cao
- **Adam**: bước riêng từng tham số · **RMSProp** · **AdamW**
- **LR scheduling**: step decay, plateau · **Warm-up**
- **Khởi tạo**: Xavier, He · **Gradient clipping**

Đổi optimizer = đổi **một dòng**

---

# Regularization
| | |
|---|---|
| L1 | đẩy trọng số về 0 |
| L2 | giữ trọng số nhỏ |
| Dropout | tắt ngẫu nhiên nơ-ron (chỉ khi train) |
| BatchNorm / LayerNorm | chuẩn hóa |
| Early stopping | dừng khi validation ngừng cải thiện |

---

# Tóm tắt
- Cross-entropy ← MLE của Bernoulli
- Khởi tạo xấu: MSE kẹt, cross-entropy thoát
- `BCELoss` + `Sigmoid` + `SGD`, ngưỡng 0.5
- Lab: tự chạy thí nghiệm MSE vs cross-entropy
