# Đáp án · Quiz Module 7: Practice Project

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | Dữ liệu gốc mất cân bằng (357 B / 212 M); cân bằng để mô hình không thiên về lớp đông |
| 2 | Giữ tỉ lệ lớp giống nhau ở tập train và test |
| 3 | Test đại diện cho dữ liệu chưa thấy; fit trên test là rò rỉ dữ liệu (data leakage) |
| 4 | Logit (điểm số thô) và nhãn chỉ số lớp kiểu `long`; không cần thêm softmax |
| 5 | 2114 |
| 6 | ≈ 0.69 (`ln 2`), vì dự đoán gần như 50/50 |
| 7 | xem giải thích |
| 8 | Overfitting; dừng sớm, regularization, mô hình nhỏ hơn, thêm dữ liệu |

## Giải thích

**Câu 1.** Nếu 63% mẫu là lớp B, mô hình đoán toàn B đã đạt 63% mà chưa học gì. Cân bằng lớp làm độ chính xác phản ánh khả năng phân biệt thật sự.

**Câu 2.** Với `stratify=y`, mỗi tập con có cùng tỉ lệ lớp như `y`. Ở đây: test có 40 mẫu mỗi lớp.

**Câu 3.** Chuẩn hóa dùng trung bình và độ lệch chuẩn. Nếu tính từ cả tập test, thông tin về dữ liệu "chưa thấy" lọt vào quá trình chuẩn bị, làm kết quả đánh giá lạc quan hơn thực tế. Đó là rò rỉ dữ liệu. Cách đúng: học thống kê từ train, áp dụng cùng thống kê đó lên test.

**Câu 4.** Đầu vào là điểm số thô shape `(N, số_lớp)`, nhãn là chỉ số lớp shape `(N,)` kiểu `long`. Softmax đã nằm bên trong `CrossEntropyLoss`; thêm nữa là sai.

**Câu 5.** `30·64 + 64 = 1984` (lớp 1) và `64·2 + 2 = 130` (lớp 2): tổng **2114**.

**Câu 6.** Hai lớp cân bằng và mô hình đoán gần như ngẫu nhiên nên xác suất mỗi lớp ≈ 0.5, loss ≈ `−ln 0.5 ≈ 0.693`. Lệch xa mốc này lúc đầu thường báo hiệu lỗi shape hoặc nhãn.

**Câu 7.** Pha train: `model.train()`, tính gradient, `backward()`, `step()`, có thể xáo trộn dữ liệu. Pha đánh giá: `model.eval()`, bọc trong `torch.no_grad()`, không `backward`, không `step`, không cần xáo trộn.

**Câu 8.** Mô hình đang overfitting: nó học thuộc đặc thù của tập train. Có thể: early stopping, giảm kích thước mạng, thêm regularization (L2, dropout), thêm dữ liệu, giảm số epoch.

## Tự chấm
7 đến 8 đúng: làm lab, rồi sang Final Project. 5 đến 6: đọc lại mục 2 và 3. Dưới 5: đọc lại cả `lesson.md`.
