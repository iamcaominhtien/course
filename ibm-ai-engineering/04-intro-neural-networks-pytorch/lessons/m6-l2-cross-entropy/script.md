# Kịch bản video · Module 6, Bài 2: Cross-entropy và huấn luyện logistic regression

Thời lượng dự kiến: ~13 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Bài trước ta dựng mô hình phân loại. Giờ phải huấn luyện nó. Chọn loss nào? Câu trả lời không phải MSE, và hôm nay ta thấy vì sao.

**[0:30 · Slide 2: MSE gặp chuyện gì]** Dữ liệu nhỏ: 20 điểm, nhãn 1 khi x lớn hơn 0.2. Khởi tạo xấu: w bằng âm 5, b bằng âm 10. Cùng optimizer, cùng khởi tạo, chỉ đổi loss. Với MSE, gradient chỉ khoảng một phần trăm nghìn, loss đứng yên ở 0.35, độ chính xác 65 phần trăm, tức là chỉ bằng đoán toàn lớp 0. Với cross-entropy, gradient khoảng âm 0.21, loss giảm từ 4.5 xuống 0.05, độ chính xác 100 phần trăm.
[Chạy code hai lần, vẽ hai đường loss.]

**[2:30 · Slide 3: Vì sao]** MSE trên xác suất sigmoid có vùng phẳng: sigmoid bão hòa thì gradient biến mất, gradient descent không biết đi đâu. Cross-entropy không bị vậy.

**[3:30 · Slide 4: Bernoulli và likelihood]** Để có loss phù hợp, quay về xác suất. Bernoulli: hai kết quả, một tham số theta. Đồng xu theta bằng 0.2: ngửa, ngửa, sấp có khả năng 0.2 nhân 0.2 nhân 0.8 bằng 0.032. Đó là likelihood.

**[5:00 · Slide 5: MLE và log]** MLE chọn tham số làm likelihood lớn nhất. Nhân nhiều số nhỏ thì khó tính, nên lấy log: tích thành tổng, và cực đại không đổi.

**[6:00 · Slide 6: Cross-entropy]** Với logistic regression, mỗi nhãn là Bernoulli với theta bằng p. Log likelihood đổi dấu chính là cross-entropy: trừ y log p cộng một trừ y nhân log một trừ p. Nhãn 1 thì chỉ log p tính, nhãn 0 thì chỉ log một trừ p tính.
[Chạy code: BCELoss với p 0.9, 0.1, 0.8 và y 1, 1, 0 ra 1.339; tính tay bằng log.]

**[7:30 · Slide 7: Phạt tự tin mà sai]** Nhãn 1, đoán 0.9: loss 0.105. Đoán 0.1: loss 2.30, gấp hơn hai mươi lần. Trơn, phạt mạnh, có ý nghĩa xác suất: lý do cross-entropy là loss chuẩn.

**[8:30 · Slide 8: Huấn luyện trong PyTorch]** Bốn bước: Linear cộng Sigmoid, BCELoss, SGD, vòng lặp quen thuộc. Xong thì ngưỡng 0.5 để ra nhãn, rồi tính accuracy.

**[9:45 · Slide 9: Optimizer nâng cao]** Adam: bước riêng cho từng tham số, mặc định tốt. RMSProp: chuẩn hóa theo bình phương gradient. AdamW: tách weight decay. Scheduling giảm learning rate dần. Warm-up tăng dần lúc đầu. Khởi tạo Xavier hoặc He. Gradient clipping chặn gradient quá lớn. Đổi optimizer chỉ một dòng code.

**[11:15 · Slide 10: Regularization]** L1 đẩy trọng số về 0. L2 giữ trọng số nhỏ. Dropout tắt ngẫu nhiên nơ-ron lúc train. Batch norm và layer norm chuẩn hóa. Early stopping dừng khi validation ngừng cải thiện.

**[12:30 · Slide 11: Tóm tắt]** Cross-entropy cho phân loại. Khởi tạo xấu: MSE kẹt, cross-entropy thoát. Làm lab: bạn sẽ tự chạy thí nghiệm này.
