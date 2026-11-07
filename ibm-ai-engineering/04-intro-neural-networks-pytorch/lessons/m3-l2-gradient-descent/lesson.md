# Module 3 · Bài 2: Gradient descent và huấn luyện hồi quy tuyến tính

> Nguồn: Module "Applying Linear Regression and Gradient Descent", Lesson 2 (3 video, 2 reading dạng widget, 1 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học. Hai reading dạng widget không đọc được qua trang web, nên nội dung tương ứng lấy từ video.
> Thời gian đọc: ~30 phút. Làm lab: ~1 giờ 15 phút. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Giải thích gradient descent làm gì, và đọc được công thức cập nhật tham số.
2. Phân biệt **loss** (một mẫu) với **cost** (cả tập dữ liệu).
3. Giải thích ảnh hưởng của **learning rate** và các tiêu chí dừng.
4. Viết vòng lặp huấn luyện hồi quy tuyến tính thủ công bằng autograd.
5. Phân biệt batch, stochastic và mini-batch gradient descent (ở mức khái niệm).

## 1. Gradient descent
Hình dung bạn đứng trên sườn núi mù sương, muốn xuống thung lũng. Bạn chỉ thấy độ dốc ngay dưới chân, nên bước một bước nhỏ theo hướng dốc xuống, rồi lặp lại. **Gradient descent** làm đúng như vậy với hàm loss: điều chỉnh tham số từng bước nhỏ theo hướng làm loss giảm.

Thuật toán:
1. Khởi tạo tham số (ngẫu nhiên hoặc giá trị chọn trước).
2. Tính dự đoán và **loss** với tham số hiện tại.
3. Tính **gradient** của loss theo từng tham số (autograd lo việc này).
4. Cập nhật tham số **ngược hướng gradient**.
5. Lặp lại đến khi hội tụ.

Vì sao ngược hướng? Gradient là vector các đạo hàm riêng, chỉ hướng loss **tăng nhanh nhất**. Muốn giảm loss thì đi theo hướng ngược lại. Độ lớn gradient cho biết loss đổi nhanh đến đâu.

> **tham số ← tham số − learning_rate × gradient**
>
> Với hồi quy tuyến tính: `w ← w − lr · ∂loss/∂w`, `b ← b − lr · ∂loss/∂b`.

## 2. Learning rate
**Learning rate** (`lr`) là siêu tham số quyết định mỗi bước đi xa bao nhiêu.
- **Quá nhỏ:** hội tụ rất chậm, cần nhiều vòng lặp.
- **Quá lớn:** bước nhảy vượt qua điểm cực tiểu, loss dao động hoặc **tăng vọt (phân kỳ)**.
- **Vừa phải:** tiến nhanh mà vẫn ổn định.

Ví dụ thật với dữ liệu của lab (tham số khởi đầu `w=-15`, `b=-10`, 15 vòng lặp): `lr = 0.1` loss còn khoảng 0.18; `lr = 0.2` loss còn khoảng 0.008 (nhanh hơn rõ rệt); `lr = 0.4` loss **tăng** đến hàng triệu (phân kỳ).

## 3. Khi nào dừng?
Gradient descent lặp, nên cần quy tắc dừng:
- Sau một số **epoch**/vòng lặp cố định (một epoch là một lượt duyệt qua toàn bộ dữ liệu).
- Khi loss giảm rất ít giữa hai vòng liền nhau.
- Khi gradient rất nhỏ (cập nhật gần như không còn tác dụng).
- **Early stopping**: dừng khi hiệu năng trên tập validation ngừng cải thiện, để tránh overfitting (sẽ gặp ở module 4).

Quan sát đường cong loss theo vòng lặp là cách nhanh nhất để biết huấn luyện có ổn không: loss phải giảm dần rồi phẳng ra.

## 4. Loss và cost
- **Loss** đo sai số của **một** dự đoán (một mẫu). Ví dụ bình phương sai số của mẫu đó.
- **Cost** gộp sai số trên **toàn bộ** tập dữ liệu: cộng các loss lại, thường chia trung bình cho `N` để so sánh được giữa các tập khác kích thước. MSE chính là cost trung bình của loss bình phương.

Cost phụ thuộc `w` và `b`. Với hồi quy tuyến tính đơn, nó là một "mặt" trên không gian tham số, có một điểm thấp nhất. Gradient descent đi từng bước trên mặt đó, từ điểm khởi đầu tới điểm thấp nhất.

## 5. Batch gradient descent
**Batch gradient descent** tính gradient của cost bằng **tất cả** mẫu trong mỗi lần cập nhật. Năm bước:
1. Dự đoán cho mọi mẫu.
2. Tính loss từng mẫu.
3. Gộp thành cost.
4. Tính gradient của cost theo tham số.
5. Cập nhật tham số.

Lặp đến khi cost hội tụ. Vì dùng toàn bộ dữ liệu nên bước cập nhật ổn định, nhưng tốn tính toán khi dữ liệu lớn. Hai biến thể:
- **Stochastic gradient descent (SGD):** mỗi lần cập nhật dùng **một** mẫu.
- **Mini-batch gradient descent:** mỗi lần dùng một **nhóm nhỏ** mẫu. Cân bằng giữa tốc độ và độ ổn định, nên được dùng nhiều nhất trong deep learning.

Hai biến thể này là chủ đề của module 4.

## 6. Vòng lặp huấn luyện trong PyTorch
Với dữ liệu `X`, `Y` (tạo ở lab), tham số `w`, `b` có `requires_grad=True`:

```python
lr = 0.1
LOSS = []                                   # lưu loss để vẽ đường cong

def train_model(iterations):
    for epoch in range(iterations):
        Yhat = forward(X)                   # 1. dự đoán
        loss = criterion(Yhat, Y)           # 2. tính loss/cost (MSE)
        LOSS.append(loss.item())
        loss.backward()                     # 3. gradient: w.grad, b.grad
        with torch.no_grad():               # 4. cập nhật, không ghi vào đồ thị
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                      # 5. đặt gradient về 0 cho vòng sau
        b.grad.zero_()
```
Năm bước này (dự đoán, loss, backward, cập nhật, reset gradient) là **khung của mọi vòng lặp huấn luyện** trong PyTorch, kể cả mạng rất sâu. Quên bước 5 thì gradient cộng dồn (bài 2 của module 1), quên `torch.no_grad()` thì phép cập nhật bị ghi vào đồ thị tính toán.

Hai lưu ý hay gặp:
- **Gradient bùng nổ / biến mất:** gradient quá lớn làm cập nhật bất ổn; quá nhỏ làm tham số gần như không đổi. Hiếm gặp ở hồi quy tuyến tính nhưng rất quan trọng với mạng sâu.
- Sau huấn luyện, loss giảm và đường thẳng khớp dữ liệu hơn. Khi loss phẳng ra và các cập nhật rất nhỏ, mô hình đã gần tham số tối ưu.

## Tóm tắt
- Gradient descent: `tham số ← tham số − lr × gradient`, lặp đến khi hội tụ.
- Learning rate quá nhỏ thì chậm, quá lớn thì phân kỳ; theo dõi đường cong loss.
- Loss là sai số một mẫu, cost là sai số cả tập; batch GD dùng cả tập mỗi bước.
- Khung huấn luyện: dự đoán, loss, `backward()`, cập nhật dưới `no_grad`, `zero_()` gradient.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
