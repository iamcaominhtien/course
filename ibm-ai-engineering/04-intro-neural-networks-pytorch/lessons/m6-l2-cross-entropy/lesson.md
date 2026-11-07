# Module 6 · Bài 2: Cross-entropy, huấn luyện logistic regression và kỹ thuật tối ưu

> Nguồn: Module "Applying Logistic Regression for Classification", Lesson 2 (5 video, 2 lab, trong đó lab Cross Entropy của khóa không có đường link tải file). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học. Phần thí nghiệm khởi tạo xấu được chạy lại và đo số liệu thật.
> Thời gian đọc: ~45 phút. Làm lab: ~1 giờ 30 phút. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Giải thích vì sao **MSE không phù hợp** cho logistic regression, và **cross-entropy** xuất phát từ đâu (MLE).
2. Tính cross-entropy bằng tay và bằng `nn.BCELoss`.
3. Huấn luyện logistic regression trong PyTorch và đo độ chính xác (accuracy).
4. Nhận biết ảnh hưởng của khởi tạo tham số xấu khi dùng MSE so với cross-entropy.
5. Nêu tên và công dụng của Adam, AdamW, RMSProp, learning rate scheduling, warm-up, khởi tạo trọng số, gradient clipping và các kỹ thuật regularization.

## 1. Vì sao không dùng MSE cho phân loại?
Với hồi quy, MSE hợp lý. Với logistic regression, mô hình xuất xác suất qua sigmoid, và MSE giữa xác suất và nhãn (0 hoặc 1) có một điểm yếu: **mặt loss có những vùng phẳng**. Khi tham số khởi đầu khiến sigmoid bão hòa (xác suất sát 0 hoặc 1 nhưng sai), gradient gần như bằng 0, và gradient descent **không biết nên đi hướng nào**: huấn luyện chậm hoặc đứng yên.

Thí nghiệm thật (20 điểm dữ liệu `x ∈ [−1, 1)`, nhãn 1 khi `x > 0.2`; khởi tạo xấu `w = −5`, `b = −10`; `lr = 2`, batch 3, 100 epoch):
| Loss | Gradient theo `w` lúc đầu | Loss đầu → cuối | Độ chính xác |
|---|---|---|---|
| MSE | ≈ −0.00001 | 0.35 → 0.35 (không đổi) | **65%** (chỉ bằng đoán toàn lớp 0) |
| Cross-entropy | ≈ −0.21 | 4.55 → 0.05 | **100%** |

Cùng dữ liệu, cùng khởi tạo, cùng optimizer; chỉ khác hàm loss. Với MSE gradient quá nhỏ nên mô hình kẹt. Với cross-entropy gradient vẫn đủ lớn nên mô hình tự thoát và học đúng.

Ngược lại, với khởi tạo tốt (ví dụ `w = 0`, `b = 0`) MSE vẫn học được (độ chính xác 100%). Nghĩa là vấn đề của MSE là **nhạy cảm với khởi tạo và có vùng gradient biến mất**, chứ không phải lúc nào cũng thất bại.

## 2. Bernoulli, likelihood và MLE
Để có hàm loss phù hợp, ta quay về xác suất.

**Phân phối Bernoulli** mô hình hóa thí nghiệm có hai kết quả (1 và 0) bằng một tham số `θ`: `P(1) = θ`, `P(0) = 1 − θ`. Ví dụ đồng xu lệch có `θ = 0.2` (xác suất ngửa).

**Likelihood** (khả năng) của dữ liệu quan sát, với `θ` cho trước, là xác suất để quan sát đúng dữ liệu đó. Với các lần tung độc lập, nhân xác suất từng lần. Chuỗi "ngửa, ngửa, sấp" với `θ = 0.2`: `0.2 × 0.2 × 0.8 = 0.032`.

**MLE (Maximum Likelihood Estimation):** chọn `θ` sao cho likelihood **lớn nhất**, tức là tham số giải thích dữ liệu tốt nhất.

Nhân nhiều xác suất nhỏ cho số cực nhỏ, khó tính số học; nên dùng **log-likelihood**: logarit biến tích thành tổng, và vì log là hàm tăng nên `θ` làm log-likelihood lớn nhất cũng làm likelihood lớn nhất.

## 3. Cross-entropy
Trong logistic regression, mỗi nhãn `yᵢ ∈ {0, 1}` được coi là Bernoulli với `θ = pᵢ` (xác suất mô hình dự đoán). Likelihood của cả tập dữ liệu:
> `L = Π pᵢ^yᵢ · (1 − pᵢ)^(1 − yᵢ)`

Lấy log, rồi đổi dấu để thành thứ ta **cực tiểu hóa**, ta được **cross-entropy** (còn gọi binary cross-entropy, log loss):

> **loss = − (1/N) · Σ [ yᵢ · log(pᵢ) + (1 − yᵢ) · log(1 − pᵢ) ]**

(Video viết dạng tổng; PyTorch lấy trung bình trên các mẫu.)

Cách đọc: nếu `y = 1` chỉ số hạng `−log(p)` có tác dụng (muốn `p` gần 1); nếu `y = 0` chỉ số hạng `−log(1 − p)` có tác dụng (muốn `p` gần 0).

**Phạt rất nặng khi tự tin mà sai.** Với `y = 1`:
| Dự đoán `p` | Loss |
|---|---|
| 0.9 (đúng, tự tin) | ≈ 0.105 |
| 0.1 (sai, tự tin) | ≈ 2.303 (gấp hơn 20 lần) |

Ví dụ tính tay với `p = [0.9, 0.1, 0.8]`, `y = [1, 1, 0]`: `−(log 0.9 + log 0.1 + log 0.2)/3 = 1.339`, và `nn.BCELoss()` cho đúng giá trị đó.

Ba ưu điểm: (1) bề mặt loss **trơn**, gradient tồn tại hầu khắp nơi nên gradient descent chạy mượt; (2) phạt mạnh lỗi tự tin, đẩy mô hình sửa nhanh; (3) có **diễn giải xác suất** rõ ràng vì suy ra từ MLE. Vì vậy cross-entropy là loss chuẩn cho phân loại, từ logistic regression đến các mạng sâu.

## 4. Huấn luyện logistic regression trong PyTorch
Bốn bước:
```python
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())   # 1. mô hình: tuyến tính + sigmoid
criterion = nn.BCELoss()                                # 2. loss: binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 3. optimizer

for epoch in range(epochs):                             # 4. vòng lặp huấn luyện quen thuộc
    for x, y in loader:
        loss = criterion(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
- `nn.BCELoss` nhận **xác suất** (đã qua sigmoid) và nhãn dạng số thực `0.0/1.0` cùng shape.
- Sau khi huấn luyện, mô hình xuất xác suất; đổi thành lớp bằng ngưỡng 0.5: `labels = (model(x) >= 0.5).float()`.
- **Độ chính xác (accuracy)** = tỉ lệ dự đoán đúng: `(labels == y).float().mean()`.

## 5. Các kỹ thuật tối ưu nâng cao (nhận biết)
Video tiếp theo điểm qua các công cụ giúp huấn luyện mạng sâu nhanh và ổn định. Ở mức nhận biết:
| Kỹ thuật | Ý chính | Hay dùng khi |
|---|---|---|
| **Adam** | giữ ước lượng trung bình của gradient và bình phương gradient, tự điều chỉnh bước cho **từng tham số** | lựa chọn mặc định, hội tụ nhanh, ít nhạy với learning rate |
| **RMSProp** | chuẩn hóa cập nhật bằng trung bình trượt của bình phương gradient, giảm dao động | mô hình chuỗi (RNN), gradient khác nhau nhiều giữa các tham số |
| **AdamW** | như Adam nhưng tách **weight decay** khỏi cập nhật gradient, regularization hiệu quả hơn | mạng lớn, transformer |
| **Learning rate scheduling** | giảm learning rate trong lúc huấn luyện: *step decay* (giảm sau số epoch cố định), *plateau* (giảm khi validation loss ngừng cải thiện), giảm dần theo thời gian | muốn học nhanh lúc đầu, tinh chỉnh lúc sau |
| **Warm-up** | bắt đầu với learning rate nhỏ rồi tăng dần | mô hình lớn, transformer, tránh cập nhật bất ổn lúc đầu |
| **Khởi tạo trọng số** | giữ cho độ lớn của activation và gradient cân bằng giữa các lớp (Xavier cho sigmoid/tanh, He/Kaiming cho ReLU) | mạng sâu, tránh gradient biến mất/bùng nổ |
| **Gradient clipping** | giới hạn độ lớn gradient trước khi cập nhật | mạng sâu, RNN, chuỗi dài |

Trong PyTorch: `optim.Adam(model.parameters(), lr=...)`, `optim.AdamW`, `optim.RMSprop`, `torch.optim.lr_scheduler.*`, `torch.nn.utils.clip_grad_norm_`. Dùng optimizer khác chỉ cần đổi một dòng; phần còn lại của vòng lặp giữ nguyên.

## 6. Regularization và khái quát hóa
**Overfitting:** tốt trên dữ liệu huấn luyện, kém trên dữ liệu mới. **Regularization** là nhóm kỹ thuật giảm overfitting:
- **L1:** phạt tổng trị tuyệt đối của trọng số; đẩy một số trọng số về đúng 0 → chọn đặc trưng tự động, mô hình đơn giản.
- **L2:** phạt tổng bình phương trọng số; giữ trọng số nhỏ và đều, mô hình mượt, ít nhạy nhiễu.
- **Dropout:** khi huấn luyện, tắt ngẫu nhiên một phần nơ-ron ở mỗi bước, để mạng không dựa quá nhiều vào vài nơ-ron. Khi đánh giá thì tắt dropout (`model.eval()`).
- **Batch normalization:** chuẩn hóa đầu vào của mỗi lớp theo thống kê của lô; huấn luyện nhanh và ổn định hơn.
- **Layer normalization:** chuẩn hóa theo các đặc trưng trong một mẫu, không phụ thuộc kích thước lô; phổ biến ở transformer và NLP.
- **Early stopping:** theo dõi validation loss, dừng khi nó ngừng cải thiện và giữ lại mô hình tốt nhất (đã gặp ở Module 4).

## Tóm tắt
- MSE có vùng gradient phẳng với sigmoid; cross-entropy (suy ra từ MLE của Bernoulli) cho bề mặt trơn và phạt nặng lỗi tự tin.
- Thí nghiệm khởi tạo xấu: MSE kẹt ở 65%, cross-entropy đạt 100%.
- Huấn luyện: `Linear` + `Sigmoid`, `BCELoss`, `SGD`; nhãn thành lớp bằng ngưỡng 0.5.
- Adam/AdamW/RMSProp, scheduling, warm-up, khởi tạo, clipping, regularization là công cụ cho mạng sâu; mỗi cái đổi chỉ một dòng code.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
