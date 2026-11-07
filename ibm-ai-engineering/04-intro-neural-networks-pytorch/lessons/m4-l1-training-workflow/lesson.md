# Module 4 · Bài 1: Huấn luyện theo kiểu PyTorch (SGD, mini-batch, optimizer, tách dữ liệu)

> Nguồn: Module "Training Linear Regression Models the PyTorch Way" (5 video, 4 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~40 phút. Làm lab: ~2 giờ. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Giải thích SGD và mini-batch gradient descent, phân biệt **batch size**, **iteration**, **epoch**.
2. Dùng `DataLoader` để chia dữ liệu thành lô.
3. Dùng `optim.SGD` và vòng lặp huấn luyện chuẩn của PyTorch: `zero_grad()`, `backward()`, `step()`.
4. Giải thích vì sao tách dữ liệu thành train / validation / test, và nhận ra overfitting.
5. Chọn siêu tham số (learning rate) dựa trên validation loss; lưu và nạp lại tham số mô hình.

## 1. Stochastic gradient descent (SGD)
Batch gradient descent dùng **toàn bộ** dữ liệu cho mỗi lần cập nhật. SGD dùng **một mẫu** mỗi lần:
1. Dự đoán cho một mẫu.
2. Tính loss của mẫu đó.
3. Tính gradient.
4. Cập nhật tham số ngay.

Lặp cho từng mẫu. Một lượt duyệt hết dữ liệu gọi là một **epoch**. Hệ quả:
- Cập nhật rất thường xuyên (mỗi epoch có `N` lần cập nhật) nên thường tiến nhanh hơn batch GD tính theo epoch, nhất là khi dữ liệu lớn.
- Mỗi lần chỉ nhìn một mẫu nên loss **dao động** mạnh giữa các bước; xu hướng chung vẫn đi xuống.

Số liệu từ lab (60 điểm dữ liệu, start `w=-15`, `b=-10`, `lr=0.1`, 10 epoch):
| Phương pháp | Số lần cập nhật | Cost cuối |
|---|---|---|
| Batch (cả 60 mẫu một lần) | 10 | ≈ 1.6 |
| SGD (1 mẫu một lần) | 600 | ≈ 0.013 |

## 2. Mini-batch gradient descent
SGD nhiễu, batch GD chậm. **Mini-batch** nằm giữa: chia dữ liệu thành lô nhỏ, mỗi lần cập nhật dùng một lô.

Ba khái niệm hay bị lẫn:
- **Batch size:** số mẫu trong một lô.
- **Iteration:** một lần cập nhật tham số (một lô).
- **Epoch:** một lượt duyệt hết dữ liệu.

`iterations mỗi epoch = số mẫu / batch size` (làm tròn lên nếu không chia hết).
Với 6 mẫu: batch size 1 → 6 iteration/epoch; 2 → 3; 3 → 2. Với 60 mẫu: batch size 5 → 12, 10 → 6, 20 → 3.

Ảnh hưởng của batch size:
| Batch nhỏ | Batch lớn |
|---|---|
| Cập nhật nhiều, nhanh | Cập nhật ít hơn |
| Loss nhiễu hơn | Loss mượt, gradient ổn định hơn |
| Tốn ít bộ nhớ | Tốn nhiều bộ nhớ |
| Có thể thoát khỏi cực tiểu địa phương | Hội tụ có thể chậm hơn tính theo epoch |

Đường đi tới cực tiểu: batch GD mượt, mini-batch hơi lắc, SGD lắc mạnh. Mini-batch được dùng nhiều nhất trong deep learning.

## 3. `DataLoader`
`DataLoader` bọc một `Dataset` và cho ra từng lô khi duyệt. Nó làm thay việc chia lô và xáo trộn.

```python
from torch.utils.data import DataLoader
loader = DataLoader(dataset=dataset, batch_size=5)
for x, y in loader:          # mỗi lần lặp là một lô: x có shape (5, 1), y có shape (5, 1)
    ...
len(loader)                  # số lô = số iteration mỗi epoch
```
- `batch_size=1` ⇒ SGD; `batch_size=len(dataset)` ⇒ batch GD.
- `shuffle=True` xáo trộn thứ tự mẫu mỗi epoch (thường bật khi huấn luyện).
- Khi số mẫu không chia hết cho batch size, lô cuối nhỏ hơn. `drop_last=True` bỏ lô cuối đó. Ví dụ 60 mẫu, batch 7: 9 lô (lô cuối 4 mẫu), hoặc 8 lô nếu `drop_last=True`.

Khi dùng `DataLoader`, vòng lặp huấn luyện có hai tầng: epoch (ngoài) và lô (trong).

## 4. Optimizer
Cập nhật tham số bằng tay (`w -= lr * w.grad`) ổn với một vài tham số nhưng không mở rộng được. PyTorch có sẵn **optimizer** (`torch.optim`): SGD, Adam, RMSprop... Optimizer giữ danh sách tham số và áp dụng quy tắc cập nhật.

```python
from torch import nn, optim
model = linear_regression(1, 1)                 # lớp nn.Module như ở Module 3
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
- `model.parameters()` cung cấp mọi tham số học được; optimizer nhớ chúng.
- `optimizer.state_dict()` cho xem cấu hình (learning rate...).

**Vòng lặp huấn luyện chuẩn** (nhớ thuộc lòng):
```python
for epoch in range(epochs):
    for x, y in trainloader:
        yhat = model(x)               # 1. forward
        loss = criterion(yhat, y)     # 2. loss
        optimizer.zero_grad()         # 3. xóa gradient cũ
        loss.backward()               # 4. tính gradient
        optimizer.step()              # 5. cập nhật tham số
```
`loss.backward()` ghi gradient vào `.grad` của từng tham số, còn `optimizer.step()` dùng chúng để cập nhật. Quên `zero_grad()` thì gradient cộng dồn. Thứ tự `zero_grad()` trước `backward()` hay ngay đầu vòng đều được, miễn là xóa **trước** khi tính gradient mới.

Đặt tham số khởi đầu thủ công (để làm thí nghiệm giống nhau): `model.state_dict()['linear.weight'][0] = -15` rồi tương tự cho bias.

## 5. Tách dữ liệu: train, validation, test
Điều quan trọng không phải mô hình khớp dữ liệu đã thấy đến đâu, mà là nó dự đoán dữ liệu **chưa thấy** tốt đến đâu. Vì vậy chia dữ liệu thành ba phần:
| Tập | Vai trò |
|---|---|
| **Training** | học tham số (`w`, `b`) |
| **Validation** | trong lúc phát triển: đánh giá, chọn siêu tham số, chọn mô hình tốt nhất |
| **Test** | chỉ dùng **một lần cuối** để ước lượng hiệu năng thật của mô hình đã chọn |

### Overfitting
Mô hình học "thuộc" cả nhiễu và các điểm bất thường trong tập huấn luyện nên dự đoán dữ liệu mới kém. Dấu hiệu rõ nhất: **training loss tiếp tục giảm trong khi validation loss bắt đầu tăng**. Mô hình tốt nhất thường được chọn theo validation loss, không theo training loss.

### Siêu tham số (hyperparameter)
Tham số (`w`, `b`) do huấn luyện tìm ra. **Siêu tham số** do bạn chọn trước và điều khiển quá trình huấn luyện: learning rate, batch size, số epoch. Cách chọn:
1. Huấn luyện nhiều mô hình với các giá trị siêu tham số khác nhau.
2. Đo validation loss của từng mô hình.
3. Chọn mô hình có validation loss thấp nhất.
4. Chỉ sau đó mới đánh giá **một lần** trên tập test.

Ví dụ trong lab: dữ liệu huấn luyện cố tình có điểm ngoại lai; thử `lr ∈ {0.0001, 0.001, 0.01, 0.1}`. Mô hình có training loss thấp nhất **không** phải mô hình có validation loss thấp nhất. Chọn theo training loss sẽ chọn sai.

### Khi đánh giá, đừng cập nhật mô hình
Đánh giá trên validation/test chỉ cần dự đoán và tính loss, không `backward()`, không `step()`. Bọc trong `torch.no_grad()` để khỏi tốn bộ nhớ theo dõi gradient:
```python
with torch.no_grad():
    val_loss = criterion(model(val_data.x), val_data.y)
```

## 6. Lưu và nạp mô hình
Sau khi chọn mô hình tốt nhất, lưu tham số để dùng lại mà không cần huấn luyện lại:
```python
torch.save(model.state_dict(), "best.pt")        # lưu dictionary tham số

new_model = linear_regression(1, 1)              # tạo mô hình cùng kiến trúc
new_model.load_state_dict(torch.load("best.pt")) # nạp tham số
new_model.eval()                                 # chế độ đánh giá
```
Mô hình mới phải **cùng kiến trúc** với mô hình đã lưu. `.eval()` chuyển sang chế độ suy luận (quan trọng với các lớp như dropout, batch norm ở các khóa sau).

## Tóm tắt
- SGD dùng 1 mẫu, batch dùng cả tập, mini-batch dùng một lô; số iteration mỗi epoch = số mẫu / batch size.
- `DataLoader` lo việc chia lô và xáo trộn; vòng lặp có hai tầng epoch và lô.
- Vòng lặp chuẩn: forward, loss, `zero_grad()`, `backward()`, `step()` với `optim.SGD(model.parameters(), lr)`.
- Train / validation / test: học, chọn siêu tham số, đánh giá cuối. Chọn mô hình theo validation loss.
- `torch.save(model.state_dict(), ...)` và `load_state_dict(...)` để tái sử dụng mô hình.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
