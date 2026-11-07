---
marp: true
theme: default
paginate: true
title: Gradient descent và huấn luyện hồi quy tuyến tính
---

# Gradient descent
Module 3 · Bài 2 · Introduction to Neural Networks and PyTorch

---

# Hình dung: xuống núi trong sương
Chỉ thấy độ dốc dưới chân → bước nhỏ xuống dốc → lặp lại

Gradient descent: chỉnh tham số từng bước nhỏ **theo hướng làm loss giảm**

---

# Thuật toán
1. Khởi tạo tham số
2. Tính loss
3. Tính gradient
4. Cập nhật **ngược** hướng gradient
5. Lặp đến khi hội tụ

**tham số ← tham số − lr × gradient**

---

# Learning rate
| lr | 15 vòng, từ w=−15, b=−10 | |
|---|---|---|
| 0.1 | loss ≈ 0.18 | chậm hơn |
| 0.2 | loss ≈ 0.008 | nhanh |
| 0.4 | loss → hàng triệu | **phân kỳ** |

Quá nhỏ: chậm · Quá lớn: vượt đáy, phân kỳ

---

# Khi nào dừng?
- Đủ số epoch
- Loss giảm rất ít
- Gradient rất nhỏ
- Early stopping (validation)

👉 Luôn xem đường cong loss

---

# Loss và cost
- **Loss**: sai số của **một** mẫu
- **Cost**: sai số của **cả tập** (trung bình các loss)
- Cost phụ thuộc `w`, `b` → "mặt" có một đáy

---

# Batch gradient descent
Dự đoán mọi mẫu → loss từng mẫu → cost → gradient → cập nhật

| Biến thể | Mỗi lần cập nhật dùng |
|---|---|
| Batch | cả tập |
| Stochastic | 1 mẫu |
| Mini-batch | 1 nhóm nhỏ |

---

# Vòng lặp huấn luyện
```python
for epoch in range(iterations):
    Yhat = forward(X)            # 1. dự đoán
    loss = criterion(Yhat, Y)    # 2. loss
    loss.backward()              # 3. gradient
    with torch.no_grad():        # 4. cập nhật
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()   # 5. reset
```

---

# Theo dõi huấn luyện
- Loss giảm dần rồi phẳng → hội tụ
- Đường thẳng khớp dữ liệu hơn sau mỗi vòng
- Gradient quá lớn/nhỏ → bất ổn / không học

---

# Tóm tắt
- `tham số ← tham số − lr × gradient`
- Learning rate vừa phải
- Khung 5 bước dùng cho mọi mô hình
- Tiếp theo: lab thử nhiều learning rate
