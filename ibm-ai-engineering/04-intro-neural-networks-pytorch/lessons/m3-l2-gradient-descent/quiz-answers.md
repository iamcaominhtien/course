# Đáp án · Quiz Module 3, Bài 2

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | Gradient chỉ hướng loss tăng nhanh nhất; gradient descent đi ngược lại |
| 2 | `w ← w − lr · ∂loss/∂w` |
| 3 | `5.0 − 0.1 × 2.0 = 4.8` |
| 4 | Quá nhỏ: hội tụ rất chậm; quá lớn: dao động hoặc phân kỳ |
| 5 | Đủ số epoch; loss giảm rất ít; gradient rất nhỏ (hoặc early stopping) |
| 6 | Loss: một mẫu; cost: tổng hợp (trung bình) trên cả tập |
| 7 | Dùng cả tập / 1 mẫu / 1 nhóm nhỏ mỗi lần cập nhật; mini-batch phổ biến nhất |
| 8 | Đặt gradient về 0 (`zero_()`); quên thì gradient cộng dồn |
| 9 | Để phép cập nhật không bị ghi vào đồ thị tính toán |
| 10 | Learning rate quá lớn; giảm learning rate |

## Giải thích

**Câu 1.** Gradient là vector đạo hàm riêng, chỉ hướng làm loss tăng nhanh nhất. Muốn loss giảm thì đi ngược hướng đó.

**Câu 3.** Công thức: `w_mới = w − lr × gradient = 5.0 − 0.1 × 2.0 = 4.8`. Gradient dương nghĩa là tăng `w` làm loss tăng, nên ta giảm `w`.

**Câu 4.** Learning rate quá nhỏ: mỗi bước quá ngắn, cần rất nhiều vòng lặp mới tới đáy. Quá lớn: bước nhảy vượt qua đáy, loss dao động quanh đáy hoặc càng lúc càng lớn (phân kỳ).

**Câu 5.** Ba tiêu chí: (1) số epoch cố định; (2) loss giảm không đáng kể giữa hai vòng; (3) gradient rất nhỏ. Ngoài ra còn early stopping dựa trên tập validation để tránh overfitting.

**Câu 6.** Loss đo sai số của một dự đoán (một mẫu). Cost gộp loss của mọi mẫu, thường lấy trung bình để so sánh được giữa các tập khác kích thước.

**Câu 7.** Batch dùng toàn bộ mẫu cho mỗi lần cập nhật (ổn định, tốn tính toán). Stochastic dùng một mẫu (nhanh nhưng nhiễu). Mini-batch dùng nhóm nhỏ, cân bằng hai điều đó nên được dùng nhiều nhất.

**Câu 8.** Đặt gradient về 0 (`w.grad.zero_()` hoặc tương tự). Nếu quên, `backward()` ở vòng sau cộng dồn vào gradient cũ, bước cập nhật sai.

**Câu 9.** Nếu không có `no_grad`, PyTorch ghi cả phép cập nhật vào đồ thị tính toán và thay đổi tham số lá sai cách (thường báo lỗi in-place trên tensor cần gradient). Ta chỉ muốn đổi giá trị tham số, không muốn theo dõi phép đó.

**Câu 10.** Learning rate quá lớn khiến mỗi bước vượt qua đáy và loss tăng. Việc thử đầu tiên: giảm learning rate (ví dụ chia cho 2 hoặc 10) rồi xem loss có giảm không.

## Tự chấm
9 đến 10 đúng: sang lab. 6 đến 8: đọc lại mục 1, 2, 6. Dưới 6: đọc lại cả `lesson.md`.
