# Kịch bản video · Bài 2: Tensor 2D và đạo hàm tự động

Thời lượng dự kiến: ~11 phút. Mỗi phần ứng với một slide hoặc nhóm slide trong [`slides/slides.md`](slides/slides.md).

---

**[0:00 · Slide 1: Tiêu đề]**
Ở bài trước ta làm quen vector. Hôm nay có hai việc: nâng lên hai chiều, và gặp thứ làm nên sức mạnh của PyTorch: đạo hàm tự động.

**[0:25 · Slide 2: Tensor 2D là một bảng]**
Tensor 2D là một lưới có hàng và cột. Trong học máy, mỗi hàng là một mẫu, mỗi cột là một đặc trưng. Ảnh xám cũng là bảng: mỗi ô là độ sáng một pixel. Tạo bằng list lồng nhau, mỗi list con là một hàng.
[Chạy code: tạo A 3x3, in ndimension, shape, numel.]

**[1:20 · Slide 3: Index và slice]**
Hai chỉ số, hàng trước cột sau, đếm từ 0. `A[1, 2]` là hàng thứ hai, cột thứ ba. `A[1, :]` lấy cả một hàng, `A[:, 2]` lấy cả một cột, `A[:, 0:2]` lấy ma trận con.
Một bẫy kinh điển: `A[1:3][2]` khác `A[1:3, 2]`. Vế đầu cắt hàng rồi mới index trên kết quả. Muốn chọn cả hàng lẫn cột, đặt hai chỉ số trong cùng một cặp ngoặc.
[Chạy code cả hai cách để thấy khác nhau.]

**[2:40 · Slide 4: Cộng, nhân vô hướng, Hadamard]**
Cộng và nhân vô hướng làm trên từng phần tử. Dấu `*` giữa hai tensor là Hadamard, nhân từng cặp cùng vị trí. Đừng nhầm với nhân ma trận.

**[3:30 · Slide 5: Nhân ma trận]**
Nhân ma trận là hàng nhân cột. Ô (i, j) của kết quả là dot giữa hàng i bên trái và cột j bên phải. Điều kiện: số cột bên trái bằng số hàng bên phải. Hai-nhân-ba nhân ba-nhân-hai ra hai-nhân-hai. Dùng toán tử `@`.
[Chạy code, rồi chạy một ví dụ lệch kích thước để thấy lỗi.]
Đây chính là phép tính trong một lớp fully connected: đầu vào nhân ma trận trọng số.

**[4:50 · Slide 6: Lên 3D]**
Thêm một chiều là có tensor 3D. Ảnh màu là cao, rộng, ba kênh. Lưu ý: PyTorch thích kênh đứng đầu, thư viện ảnh thường để kênh ở cuối. Đổi trục bằng `permute`.

**[5:30 · Slide 7: Vì sao cần đạo hàm?]**
Sang phần hai. Huấn luyện là chỉnh tham số để loss nhỏ đi. Muốn biết chỉnh tham số nào, hướng nào, ta cần biết loss đổi bao nhiêu khi tham số đổi một chút. Đó là đạo hàm. Mạng có hàng triệu tham số, nên máy phải tự làm. Đó là autograd.

**[6:20 · Slide 8: Autograd hoạt động thế nào]**
Bước một: tạo tensor với `requires_grad=True`. Bước hai: PyTorch ghi lại các phép toán thành một đồ thị. Bước ba: gọi `backward()`. Bước bốn: đọc kết quả ở `.grad`.
[Chạy: x = 2, y = x² + 2x + 1, backward, in x.grad, ra 6.] Kiểm tra bằng tay: đạo hàm là 2x cộng 2, tại x bằng 2 là 6. Khớp.

**[7:40 · Slide 9: Đồ thị tính toán]**
[Hiện hình đồ thị: x → x² và 3x → cộng → y.] Mỗi nút là một phép toán. Khi `backward`, PyTorch đi ngược từ y về x.

**[8:20 · Slide 10: Đạo hàm riêng]**
Hàm nhiều biến: đạo hàm riêng theo từng biến, giữ các biến khác cố định. Với f bằng u nhân v cộng u bình, theo u là v cộng 2u, theo v là u. PyTorch cho cả hai chỉ với một lần `backward`.

**[9:10 · Slide 11: Chain rule]**
Mạng nhiều lớp là hàm lồng hàm. Chain rule: đạo hàm bằng tích các đạo hàm thành phần. Autograd tự áp dụng khi đi ngược, nên sâu bao nhiêu lớp cũng chạy.

**[9:50 · Slide 12: Ba điều cần nhớ]**
Một: `backward` cần kết quả là một số, nếu là vector thì cộng gộp bằng `sum` trước. Hai: gradient cộng dồn, nên đặt lại về 0 mỗi lượt. Ba: muốn vẽ đồ thị thì `detach` rồi mới `numpy`.

**[10:30 · Slide 13: Tóm tắt]**
Tensor 2D: hàng là mẫu, cột là đặc trưng. `*` là Hadamard, `@` là nhân ma trận. Autograd ghi đồ thị, `backward` đi ngược, `.grad` giữ kết quả. Hai lab đang chờ bạn. Hẹn gặp ở module sau.
