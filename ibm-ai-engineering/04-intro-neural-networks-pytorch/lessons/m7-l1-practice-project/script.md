# Kịch bản video · Module 7: Practice Project, ung thư vú

Thời lượng dự kiến: ~9 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Đây là dự án thực hành cuối: một quy trình phân loại hoàn chỉnh trên dữ liệu y khoa thật. Mọi mảnh ghép bạn đã học.

**[0:25 · Slide 2: Dữ liệu]** Breast Cancer Wisconsin: 569 mẫu, 30 đặc trưng về nhân tế bào, nhãn lành tính hoặc ác tính. Phân bố 357 lành tính, 212 ác tính: mất cân bằng. Ta lấy 200 mẫu mỗi lớp để cân bằng.
[Chạy code: fetch_ucirepo, value_counts, sample 200.]

**[1:45 · Slide 3: Tiền xử lý]** Nhãn thành số. Chia 80/20 với stratify để giữ tỉ lệ lớp. Chuẩn hóa: fit trên train, chỉ transform trên test. Nếu fit trên test là rò rỉ dữ liệu. Tensor: đặc trưng float32, nhãn long. Rồi TensorDataset và DataLoader.
[Chạy code: in trung bình cột của train và test để thấy khác nhau.]

**[3:30 · Slide 4: Mạng nơ-ron]** Hai lớp Linear, giữa là ReLU. Đầu ra hai điểm số, không có sigmoid. Hơn 2000 tham số. Đây là mạng nơ-ron đầu tiên của bạn.

**[4:30 · Slide 5: CrossEntropyLoss]** Nhận điểm số thô, tự softmax bên trong, nhãn là chỉ số lớp. Chạy với bao nhiêu lớp cũng được: đổi output_units. Mô hình chưa huấn luyện có loss khoảng 0.69, là ln 2.

**[5:30 · Slide 6: Huấn luyện hai pha]** Mỗi epoch: pha train gọi model.train, cập nhật theo lô. Pha đánh giá gọi model.eval, no_grad, chỉ tính loss. Lưu hai loss và vẽ.
[Chạy code 10 epoch; vẽ hai đường loss.]

**[7:00 · Slide 7: Đọc đường loss]** Hai đường giảm cùng nhau: tốt. Train giảm, test tăng: overfitting. Độ chính xác test trên 93 phần trăm.

**[7:45 · Slide 8: Ba bài luyện tập]** Đổi sang SGD với momentum. Giảm lớp ẩn xuống 16. Thử Iris với ba lớp.

**[8:30 · Slide 9: Tóm tắt]** Cân bằng, chuẩn hóa đúng cách, mạng một lớp ẩn, hai pha huấn luyện. Bạn sẵn sàng cho Final Project.
