# Đáp án · Quiz Module 6, Bài 2

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | Mặt loss có vùng phẳng; gradient biến mất khi sigmoid bão hòa |
| 2 | `P(1) = 0.2`, `P(0) = 0.8`; likelihood `= 0.2 × 0.2 × 0.8 = 0.032` |
| 3 | Chọn tham số làm likelihood lớn nhất; log biến tích thành tổng, ổn định số học |
| 4 | `−[y·log p + (1−y)·log(1−p)]` |
| 5 | `p = 0.9`: ≈ 0.105; `p = 0.1`: ≈ 2.303 (gấp > 20 lần) |
| 6 | Xác suất (sau sigmoid); nhãn dạng float 0.0/1.0 |
| 7 | `>=`; `mean()` |
| 8 | MSE: gradient ≈ 0 nên kẹt; cross-entropy: gradient đủ lớn nên thoát |
| 9 | xem giải thích |
| 10 | L1 đẩy trọng số về đúng 0; L2 làm trọng số nhỏ và đều |
| 11 | Train: tắt ngẫu nhiên nơ-ron; eval: dùng tất cả; `eval()` tắt dropout |
| 12 | Batch norm: theo lô; layer norm: theo đặc trưng của một mẫu; layer norm ở transformer |

## Giải thích

**Câu 1.** MSE trên đầu ra sigmoid có những vùng loss gần như phẳng. Ở đó gradient ≈ 0, gradient descent không biết hướng đi, huấn luyện chậm hoặc đứng. Đó là lý do ít dùng MSE cho phân loại.

**Câu 3.** MLE chọn tham số khiến dữ liệu quan sát có khả năng lớn nhất. Likelihood là tích nhiều xác suất nhỏ nên rất nhỏ và khó tính số học; log biến tích thành tổng. Vì log là hàm tăng, tham số làm log-likelihood lớn nhất cũng làm likelihood lớn nhất.

**Câu 4.** Một mẫu: `loss = −[y·log(p) + (1−y)·log(1−p)]`; trung bình trên `N` mẫu cho loss của cả tập. Nó chính là âm của log-likelihood Bernoulli.

**Câu 5.** `y = 1` nên `loss = −log p`. `p = 0.9`: `−(−0.105) = 0.105`. `p = 0.1`: `−(−2.303) = 2.303`. Dự đoán tự tin mà sai bị phạt nặng hơn hơn 20 lần.

**Câu 6.** `nn.BCELoss` mong xác suất trong [0, 1] (đầu ra của `Sigmoid`), và nhãn kiểu float cùng shape với dự đoán. (Biến thể `nn.BCEWithLogitsLoss` nhận logit, gộp sẵn sigmoid; sẽ gặp khi cần.)

**Câu 7.** `labels = (model(x) >= 0.5).float()`; `accuracy = (labels == y).float().mean()`.

**Câu 8.** Với `w = −5`, `b = −10`, sigmoid bão hòa về 0 cho cả những mẫu thuộc lớp 1. Dưới MSE, gradient ≈ 10⁻⁵ nên tham số gần như không đổi; mô hình đoán toàn lớp 0 (65% = 13/20 mẫu lớp 0). Dưới cross-entropy, lỗi tự tin bị phạt nặng và gradient ≈ −0.21 nên mô hình sửa nhanh và đạt 100%.

**Câu 9.** Adam: học với bước thích ứng cho từng tham số, hội tụ nhanh, ít nhạy với learning rate. AdamW: Adam với weight decay tách riêng, regularization tốt hơn. LR scheduling: đổi learning rate trong lúc train để học nhanh lúc đầu, tinh chỉnh lúc sau. Warm-up: tăng learning rate dần từ nhỏ ở đầu quá trình để tránh cập nhật bất ổn. Gradient clipping: giới hạn độ lớn gradient để tránh bùng nổ.

**Câu 10.** L1 phạt tổng trị tuyệt đối trọng số, nên một số trọng số về đúng 0 (chọn đặc trưng, mô hình thưa). L2 phạt tổng bình phương, làm mọi trọng số nhỏ nhưng thường khác 0, mô hình mượt.

**Câu 11.** Khi huấn luyện, dropout ngẫu nhiên tắt một phần nơ-ron ở mỗi bước, bắt mạng không dựa vào vài nơ-ron. Khi đánh giá, mọi nơ-ron hoạt động. `model.eval()` chuyển mô hình sang chế độ đánh giá và tắt dropout (batch norm cũng đổi cách hoạt động).

**Câu 12.** Batch norm chuẩn hóa theo thống kê của cả lô, nên phụ thuộc kích thước lô. Layer norm chuẩn hóa trên các đặc trưng của riêng từng mẫu, không phụ thuộc kích thước lô, hợp với lô nhỏ và chuỗi. Transformer dùng layer norm.

## Tự chấm
11 đến 12 đúng: sang lab. 8 đến 10: đọc lại mục 1, 3, 4. Dưới 8: đọc lại cả `lesson.md`.
