# Quiz · Module 3, Bài 2: Gradient descent và huấn luyện

10 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** Gradient của loss chỉ hướng nào, và gradient descent đi theo hướng nào?

**Câu 2.** Viết công thức cập nhật tham số `w` của gradient descent.

**Câu 3.** `w = 5.0`, gradient của loss theo `w` là `2.0`, learning rate `0.1`. Sau một bước cập nhật, `w` bằng bao nhiêu?

**Câu 4.** Learning rate quá nhỏ và quá lớn gây ra hậu quả gì?

**Câu 5.** Nêu ba tiêu chí dừng vòng lặp huấn luyện.

**Câu 6.** Phân biệt loss và cost.

**Câu 7.** Batch, stochastic và mini-batch gradient descent khác nhau ở điểm nào? Loại nào phổ biến nhất trong deep learning?

**Câu 8.** Một vòng lặp huấn luyện PyTorch có đủ năm bước: dự đoán, tính loss, `loss.backward()`, cập nhật tham số, và một bước còn thiếu. Bước còn thiếu là gì, và quên nó thì sao?

**Câu 9.** Vì sao phép cập nhật `w -= lr * w.grad` phải đặt trong `with torch.no_grad():`?

**Câu 10.** Bạn huấn luyện và thấy loss **tăng dần** qua các vòng, thậm chí thành số rất lớn. Nguyên nhân phổ biến nhất là gì và bạn thử làm gì đầu tiên?
