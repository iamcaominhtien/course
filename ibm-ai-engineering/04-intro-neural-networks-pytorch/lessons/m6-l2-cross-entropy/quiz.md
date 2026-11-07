# Quiz · Module 6, Bài 2: Cross-entropy và huấn luyện logistic regression

12 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** Vì sao MSE thường không phải lựa chọn tốt cho logistic regression?

**Câu 2.** Phân phối Bernoulli với `θ = 0.2` mô tả điều gì? Tính likelihood của chuỗi "ngửa, ngửa, sấp" (ngửa = 1) với `θ = 0.2`.

**Câu 3.** MLE là gì, và vì sao người ta dùng log-likelihood thay vì likelihood?

**Câu 4.** Viết công thức binary cross-entropy cho một mẫu có nhãn `y ∈ {0, 1}` và xác suất dự đoán `p`.

**Câu 5.** Tính cross-entropy (một mẫu) khi `y = 1` và `p = 0.9`, rồi khi `p = 0.1`. So sánh. (Cho `ln 0.9 ≈ −0.105`, `ln 0.1 ≈ −2.303`.)

**Câu 6.** `nn.BCELoss` nhận đầu vào là gì: logit (điểm số `z`) hay xác suất? Nhãn có kiểu dữ liệu gì?

**Câu 7.** Điền chỗ trống: sau khi huấn luyện, nhãn dự đoán lấy bằng `labels = (model(x) ____ 0.5).float()`, và accuracy = `(labels == y).float().____()`.

**Câu 8.** Một mô hình khởi tạo `w = −5`, `b = −10` được huấn luyện bằng MSE và kẹt ở 65% accuracy, còn dùng cross-entropy thì đạt 100%. Giải thích hiện tượng.

**Câu 9.** Nêu công dụng của: Adam, AdamW, learning rate scheduling, warm-up, gradient clipping.

**Câu 10.** Phân biệt L1 và L2 regularization về tác động lên trọng số.

**Câu 11.** Dropout hoạt động thế nào khi huấn luyện và khi đánh giá? Gọi `model.eval()` có ảnh hưởng gì?

**Câu 12.** Khác nhau giữa batch normalization và layer normalization là gì? Loại nào hay dùng ở transformer?
