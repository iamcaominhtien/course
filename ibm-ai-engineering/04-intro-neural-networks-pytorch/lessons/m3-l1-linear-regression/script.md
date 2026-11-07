# Kịch bản video · Module 3, Bài 1: Hồi quy tuyến tính, dự đoán và loss

Thời lượng dự kiến: ~10 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Hôm nay mình huấn luyện mô hình đầu tiên. Mô hình đơn giản nhất: một đường thẳng.

**[0:25 · Slide 2: Bài toán]** Có các cặp x, y: diện tích nhà và giá nhà, giờ học và điểm thi. Có đáp án sẵn, nên đây là học có giám sát. Mô hình: y mũ bằng w nhân x cộng b. W là độ dốc, b là hệ số chặn. Huấn luyện là đi tìm w và b.

**[1:30 · Slide 3: Nhiễu]** Dữ liệu thật không nằm đúng trên đường thẳng. Phần lệch gọi là nhiễu, thường giả định là Gauss: lệch nhỏ phổ biến, lệch lớn hiếm. Ta muốn bắt xu hướng chung, không khớp từng điểm.

**[2:20 · Slide 4: Dự đoán bằng tensor]** Cho w bằng 2, b bằng âm 1, cả hai bật requires_grad. Hàm forward trả w nhân x cộng b. X bằng 1 ra 1. Với x có hai hàng, ra hai dự đoán, mỗi hàng một mẫu.
[Chạy code: x shape (1,1) rồi (2,1), in kết quả và shape.]

**[3:45 · Slide 5: nn.Linear]** Không cần tự viết công thức. Linear với in_features 1, out_features 1 tự tạo weight và bias. Tham số khởi tạo ngẫu nhiên nên đặt manual_seed. Gọi model như một hàm. State_dict cho thấy tên và giá trị tham số.
[Chạy code với seed 1: w khoảng 0.515, b khoảng âm 0.441.]

**[5:15 · Slide 6: Mô hình tự viết]** Mô hình lớn hơn là nhiều lớp ghép lại, nên ta viết lớp kế thừa nn.Module. Constructor gọi super init và khai báo lớp con. Forward nói cách tính dự đoán. Gọi model x, không gọi forward trực tiếp.
[Chạy code: lớp LR, in state_dict, thấy tên linear.weight.]

**[6:45 · Slide 7: Hàm loss]** Làm sao biết dự đoán tốt hay dở? Hàm loss đổi sai số thành một con số. Với hồi quy dùng MSE: trung bình bình phương sai số. Bình phương để sai số dương âm không triệt tiêu, và phạt nặng sai số lớn.
[Chạy code: MSELoss với ví dụ 1, 2, 3 và 1, 1, 5 ra 1.667; tính tay.]

**[8:30 · Slide 8: Loss là hàm của tham số]** Đổi w, b thì dự đoán đổi, loss đổi. Nên huấn luyện là tìm w, b làm loss nhỏ nhất. Cách tìm: gradient descent, bài sau.

**[9:30 · Slide 9: Tóm tắt]** Mô hình y mũ bằng w x cộng b. Dự đoán là forward pass. Loss đo sai số. Huấn luyện là tối ưu loss theo tham số. Làm lab nhé.
