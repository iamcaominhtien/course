---
marp: true
theme: default
paginate: true
title: Practice Project — phân loại ung thư vú
---

# Practice Project
Mạng nơ-ron phân loại ung thư vú · Module 7

---

# Dữ liệu
**Breast Cancer Wisconsin (Diagnostic)**
- 569 mẫu · 30 đặc trưng · nhãn `B` / `M`
- 357 B, 212 M → **mất cân bằng**
- Lấy **200 mẫu mỗi lớp** → 400 mẫu

```python
ds = fetch_ucirepo(id=17)
```

---

# Tiền xử lý
1. Nhãn: `B → 0`, `M → 1`
2. Chia 80/20, `stratify=y`
3. `StandardScaler`: `fit_transform(train)`, **chỉ** `transform(test)`
4. Tensor: đặc trưng `float32`, nhãn `long`
5. `TensorDataset` + `DataLoader`

⚠️ `fit` trên test = **rò rỉ dữ liệu**

---

# Mạng nơ-ron một lớp ẩn
```python
class ClassificationNet(nn.Module):
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super().__init__()
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```
2114 tham số · đầu ra = 2 điểm số (chưa phải xác suất)

---

# `CrossEntropyLoss`
- Nhận **logit**, tự softmax bên trong
- Nhãn: chỉ số lớp, kiểu `long`
- Đa lớp: đổi `output_units`
- Chưa huấn luyện: loss ≈ ln 2 ≈ 0.693

`optim.Adam(model.parameters(), lr=0.001)`

---

# Vòng lặp hai pha
```
model.train()   → duyệt train_loader: zero_grad, loss, backward, step
model.eval()    → no_grad, duyệt test_loader: chỉ tính loss
```
Lưu `train_losses`, `test_losses` → vẽ

---

# Đọc đường loss
| Hiện tượng | Ý nghĩa |
|---|---|
| hai đường giảm, gần nhau | tốt |
| train ↓, test ↑ | **overfitting** |

Accuracy test ≈ **93%+** sau 10 epoch

---

# Luyện tập
1. `SGD(momentum=0.9, weight_decay=1e-4)`
2. `hidden_units = 16` (2114 → 530 tham số)
3. **Iris**: 4 đặc trưng, 3 lớp

---

# Tóm tắt
- Cân bằng · chuẩn hóa đúng cách · stratify
- Mạng 1 lớp ẩn + ReLU + CrossEntropy + Adam
- `train()` / `eval()`; so train loss với test loss
- Sẵn sàng cho **Final Project**
