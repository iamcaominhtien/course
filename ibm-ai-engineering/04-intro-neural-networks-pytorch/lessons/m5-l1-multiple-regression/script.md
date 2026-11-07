# Kịch bản video · Module 5: Nhiều đầu vào, nhiều đầu ra

Thời lượng dự kiến: ~12 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Giá nhà không chỉ phụ thuộc diện tích. Hôm nay mô hình của ta có nhiều đầu vào, rồi nhiều đầu ra.

**[0:30 · Slide 2: Nhiều đầu vào]** Với d đặc trưng, có d trọng số và một bias: y mũ bằng w dot x cộng b. Cost vẫn là MSE. Gradient descent cập nhật cả vector w.

**[1:30 · Slide 3: Shape]** Một mẫu có d đặc trưng: x là 1 nhân d, w là d nhân 1, tích là 1 nhân 1. Nhiều mẫu: X là N nhân d, mỗi hàng một mẫu. Điều kiện: số cột của x bằng số hàng của w.
[Chạy code: w bằng 2, 3; b bằng 1; x bằng 1, 2 cho 9; X ba mẫu cho 6, 9, 12.]

**[3:00 · Slide 4: nn.Linear]** Linear với 2 đầu vào, 1 đầu ra. Đưa X vào ra ba dự đoán. Chú ý: weight lưu theo dạng out nhân in, tức là 1 nhân 2, chuyển vị so với công thức tay.
[Chạy code: in weight, bias và shape.]

**[4:15 · Slide 5: Mô hình tự viết và huấn luyện]** Lớp linear_regression y hệt trước, chỉ đổi input_size. Huấn luyện: dataset 20 mẫu, hai đặc trưng, DataLoader batch 2, 100 epoch, learning rate 0.1. Với hai đặc trưng, mô hình là một mặt phẳng, huấn luyện xoay mặt phẳng cho khớp dữ liệu.
[Chạy code, vẽ đường loss giảm.]

**[5:45 · Slide 6: Bẫy đặc trưng trùng nhau]** Dữ liệu lab có hai cột giống hệt nhau. Mô hình học ra w khoảng 1.4 và 0.66, không phải 1 và 1, nhưng tổng là 2. Hai cột bằng nhau thì chỉ tổng w1 cộng w2 ảnh hưởng, nên có vô số cặp cho cùng dự đoán. Dự đoán tốt nhưng từng trọng số không đọc ra ý nghĩa.

**[7:00 · Slide 7: Nhiều đầu ra]** Mỗi đầu ra có trọng số và bias riêng, dùng chung đầu vào. W là d nhân m, bias là m, kết quả N nhân m. Mỗi cột của W là một đầu ra.
[Chạy code: Linear 1 vào, 10 ra; hai shape.]

**[8:30 · Slide 8: Huấn luyện nhiều đầu ra]** Nhãn y bây giờ là vector, N nhân m. MSELoss trung bình trên mọi phần tử, nên giảm cost là cải thiện mọi đầu ra cùng lúc. Khung huấn luyện không đổi.
[Chạy code: Linear 2 vào, 2 ra, batch 5, 100 epoch; in tổng trọng số theo từng đầu ra.]

**[10:00 · Slide 9: Xu hướng PyTorch]** Mô hình ngày nay hiếm khi huấn luyện từ đầu: transfer learning với frozen backbone hoặc fine-tuning; Hugging Face cho transformer; Lightning cho tổ chức mã; TorchScript và ONNX cho triển khai. Không cần nhớ chi tiết, các khóa sau sẽ dùng lại.

**[11:15 · Slide 10: Tóm tắt]** Shape là chìa khóa. Linear in, out cho N nhân out. Khung huấn luyện giữ nguyên. Làm lab nhé.
