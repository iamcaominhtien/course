# Kịch bản video · Module 4: Huấn luyện theo kiểu PyTorch

Thời lượng dự kiến: ~13 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Module trước bạn tự tay viết vòng lặp huấn luyện. Hôm nay ta làm nó gọn hơn và đúng cách hơn: SGD, mini-batch, optimizer, và tách dữ liệu.

**[0:35 · Slide 2: SGD]** Batch gradient descent dùng cả tập cho mỗi lần cập nhật. SGD dùng một mẫu: dự đoán, loss, gradient, cập nhật, rồi sang mẫu kế tiếp. Một lượt qua hết dữ liệu là một epoch. Cập nhật thường xuyên nên tiến nhanh, nhưng loss dao động vì mỗi lần chỉ nhìn một mẫu.
[Chạy code: so sánh batch và SGD, 10 epoch.]

**[2:15 · Slide 3: Mini-batch]** Giữa hai thái cực là mini-batch: mỗi lần cập nhật dùng một lô nhỏ. Ba từ cần phân biệt: batch size, iteration, epoch. Sáu mẫu, batch size 1 là 6 iteration mỗi epoch; batch size 2 là 3; batch size 3 là 2.

**[3:30 · Slide 4: Batch size ảnh hưởng gì]** Batch nhỏ: cập nhật nhiều, nhanh, nhưng nhiễu, tốn ít bộ nhớ. Batch lớn: mượt, ổn định, tốn bộ nhớ. Mini-batch là lựa chọn phổ biến nhất.

**[4:30 · Slide 5: DataLoader]** DataLoader bọc dataset và đưa ra từng lô. Batch size 1 là SGD, batch size bằng cả tập là batch GD. Shuffle xáo trộn mỗi epoch. Len của loader là số iteration mỗi epoch.
[Chạy code: duyệt loader, in shape của lô; thử drop_last.]

**[6:00 · Slide 6: Optimizer]** Thay vì tự trừ gradient, dùng optim.SGD với tham số của mô hình và learning rate. Vòng lặp chuẩn năm bước: forward, loss, zero_grad, backward, step. Backward tính gradient, step dùng gradient để cập nhật.
[Chạy code: train với optimizer, in tham số trước và sau.]

**[8:00 · Slide 7: Vì sao tách dữ liệu]** Điều quan trọng là dự đoán dữ liệu mới. Chia thành train để học, validation để chọn siêu tham số, test để đo cuối cùng, và chỉ đo một lần.

**[9:15 · Slide 8: Overfitting]** Mô hình học thuộc cả nhiễu. Training loss giảm tiếp, validation loss bắt đầu tăng: đó là overfitting. Chọn mô hình theo validation loss.

**[10:15 · Slide 9: Chọn learning rate]** Thử bốn learning rate. Mô hình tốt nhất trên training không phải mô hình tốt nhất trên validation. Chọn theo validation. Khi đánh giá thì bọc trong no_grad và không cập nhật.
[Chạy code: bảng train error và validation error theo từng learning rate.]

**[11:45 · Slide 10: Lưu và nạp]** Lưu state_dict ra file, tạo mô hình cùng kiến trúc, load_state_dict, rồi eval. Không cần huấn luyện lại.

**[12:30 · Slide 11: Tóm tắt]** Mini-batch, DataLoader, optimizer, train/validation/test, lưu mô hình. Làm lab: bạn sẽ chạy từng thí nghiệm này.
