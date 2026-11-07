# Quiz · Module 4: Huấn luyện theo kiểu PyTorch

12 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** SGD khác batch gradient descent ở điểm cốt lõi nào?

**Câu 2.** Phân biệt batch size, iteration và epoch.

**Câu 3.** Tập có 60 mẫu. Dùng batch size 5 thì mỗi epoch có bao nhiêu iteration? Nếu huấn luyện 10 epoch, tham số được cập nhật tổng cộng mấy lần?

**Câu 4.** `DataLoader(dataset, batch_size=7)` trên 60 mẫu cho bao nhiêu lô? Lô cuối có bao nhiêu mẫu? Với `drop_last=True` thì sao?

**Câu 5.** `batch_size=1` và `batch_size=len(dataset)` tương ứng với biến thể gradient descent nào?

**Câu 6.** So sánh batch size nhỏ và lớn về độ nhiễu của loss và bộ nhớ.

**Câu 7.** Viết (hoặc sắp xếp) đúng thứ tự năm bước trong vòng lặp huấn luyện với optimizer.

**Câu 8.** `loss.backward()` và `optimizer.step()` mỗi cái làm gì?

**Câu 9.** Vì sao cần `optimizer.zero_grad()` trong mỗi iteration?

**Câu 10.** Vai trò của tập training, validation và test. Tập nào chỉ nên dùng một lần ở cuối?

**Câu 11.** Dấu hiệu nào cho thấy overfitting khi theo dõi loss?

**Câu 12.** Bạn huấn luyện 4 mô hình với 4 learning rate. Mô hình nào bạn chọn: có training loss thấp nhất, hay có validation loss thấp nhất? Vì sao? Và để lưu mô hình đã chọn bạn gọi lệnh nào?
