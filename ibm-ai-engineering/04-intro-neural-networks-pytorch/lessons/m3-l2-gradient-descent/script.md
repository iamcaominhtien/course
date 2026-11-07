# Kịch bản video · Module 3, Bài 2: Gradient descent và huấn luyện

Thời lượng dự kiến: ~11 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Bài trước ta biết đo sai số bằng loss. Hôm nay ta làm loss nhỏ đi: gradient descent.

**[0:25 · Slide 2: Hình dung]** Bạn đứng trên núi mù sương, chỉ thấy độ dốc dưới chân. Bạn bước nhỏ xuống dốc, lặp lại, cuối cùng tới thung lũng. Gradient descent chỉnh tham số từng bước nhỏ theo hướng làm loss giảm.

**[1:15 · Slide 3: Thuật toán]** Khởi tạo tham số. Tính loss. Tính gradient. Cập nhật ngược hướng gradient. Lặp lại. Gradient chỉ hướng loss tăng nhanh nhất, nên ta đi ngược. Công thức: tham số mới bằng tham số cũ trừ learning rate nhân gradient.

**[2:30 · Slide 4: Learning rate]** Learning rate quyết định bước dài hay ngắn. Quá nhỏ: chậm. Quá lớn: nhảy qua đáy, dao động hoặc phân kỳ. Với dữ liệu của lab: 0.1 còn loss 0.18, 0.2 còn 0.008, 0.4 phân kỳ.
[Chạy code ba learning rate, vẽ ba đường loss.]

**[4:00 · Slide 5: Khi nào dừng]** Sau số epoch cố định. Khi loss giảm rất ít. Khi gradient rất nhỏ. Hoặc early stopping dựa vào validation, mình gặp ở module sau. Luôn nhìn đường cong loss.

**[5:00 · Slide 6: Loss và cost]** Loss là sai số của một mẫu. Cost là sai số của cả tập: cộng các loss rồi chia trung bình. Cost phụ thuộc w và b, tạo thành một mặt trên không gian tham số, và gradient descent đi xuống đáy mặt đó.

**[6:15 · Slide 7: Batch gradient descent]** Năm bước: dự đoán mọi mẫu, loss từng mẫu, gộp thành cost, gradient của cost, cập nhật. Dùng cả tập nên ổn định nhưng tốn khi dữ liệu lớn. Biến thể: stochastic dùng một mẫu mỗi lần, mini-batch dùng nhóm nhỏ. Module sau ta làm cả hai.

**[7:30 · Slide 8: Vòng lặp trong PyTorch]** Sinh dữ liệu: X từ âm 3 đến 3, đường f bằng X trừ 1, thêm nhiễu. Rồi năm dòng quen thuộc: dự đoán, loss, backward, cập nhật trong no_grad, đặt gradient về 0.
[Chạy code train 15 vòng, in w và b.]

**[9:15 · Slide 9: Theo dõi]** Loss giảm dần rồi phẳng: mô hình hội tụ. Đường thẳng khớp dữ liệu hơn sau mỗi vòng.
[Vẽ dữ liệu cùng đường thẳng trước và sau khi train.]

**[10:15 · Slide 10: Tóm tắt]** Công thức cập nhật. Learning rate vừa phải. Khung năm bước. Làm lab: bạn sẽ thử nhiều learning rate.
