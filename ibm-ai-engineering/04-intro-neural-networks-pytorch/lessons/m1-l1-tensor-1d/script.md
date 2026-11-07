# Kịch bản video · Bài 1: Tensor 1D trong PyTorch

Thời lượng dự kiến: ~9 phút. Mỗi phần ứng với một slide hoặc nhóm slide trong [`slides/slides.md`](slides/slides.md). Phần trong ngoặc vuông là hành động trên màn hình.

---

**[0:00 · Slide 1: Tiêu đề]**
Chào bạn. Hôm nay mình bắt đầu khóa học PyTorch bằng thứ nhỏ nhất nhưng quan trọng nhất: tensor. Hiểu chắc tensor thì mọi thứ phía sau, từ hồi quy tuyến tính đến mạng tích chập, chỉ là những cách ghép tensor lại với nhau.

**[0:30 · Slide 2: Máy tính chỉ thấy số]**
Hãy tưởng tượng bạn muốn dạy máy nhận ra con mèo trong ảnh. Máy không thấy con mèo, nó chỉ thấy một dãy số. Mạng nơ-ron nhận dãy số vào, nhân với các tham số, rồi trả số ra. Dữ liệu vào, tham số, kết quả trung gian, dự đoán: tất cả đều nằm trong tensor.

**[1:15 · Slide 3: Bức tranh các kiến trúc]**
Bạn sẽ nghe nhiều tên kiến trúc. Mình tóm tắt trong một câu mỗi loại. Mạng feedforward sâu: nhiều lớp ẩn, thông tin đi một chiều, là nền của mọi thứ. CNN: dành cho ảnh, dùng bộ lọc nhỏ quét qua ảnh để bắt cạnh, kết cấu. RNN, LSTM, GRU: dành cho chuỗi, nhớ thứ tự. Transformer: dùng attention để nhìn cả chuỗi cùng lúc. Hiện chưa cần nhớ chi tiết, chỉ cần biết: loại nào cũng chạy trên tensor.

**[2:15 · Slide 4: Scalar, vector, matrix, tensor]**
Scalar là một số. Vector là một dãy số. Matrix là một bảng số. Tensor là tên chung cho mảng n chiều. Một ảnh màu là tensor ba chiều: cao, rộng, và ba kênh màu.
[Vẽ ba hình: chấm, mũi tên, lưới.]

**[3:00 · Slide 5: Tạo tensor 1D]**
[Chạy code] `torch.tensor` với một list số nguyên: kiểu `int64`. Với list số thực: `float32`. Để ý: list toàn số nguyên cho tensor nguyên. Khi huấn luyện mô hình ta cần số thực, vì chỉ tensor số thực mới tính được gradient.

**[3:50 · Slide 6: dtype và kích thước]**
`dtype` cho biết kiểu, `size` cho biết hình dạng, `numel` đếm phần tử. Đổi kiểu bằng `.float()` hoặc `.to`. Lưu ý: lệnh đổi kiểu trả về tensor mới. Một bẫy nhỏ: gọi `.mean()` trên tensor số nguyên sẽ báo lỗi. Hãy đổi sang số thực trước.

**[4:40 · Slide 7: Index, slice và view]**
Index đếm từ 0. Slice `1:4` lấy index 1, 2, 3, không lấy 4. Cho nhiều vị trí cùng lúc: đưa một list index. Và nhớ rằng slice là một view, sửa nó thì tensor gốc đổi. `view` đổi hình dạng mà không sao chép, ví dụ biến vector thành cột bằng `view(-1, 1)`.

**[5:40 · Slide 8: NumPy và Pandas]**
`from_numpy` và `.numpy()` dùng chung bộ nhớ. Mình sửa mảng NumPy,
[Chạy code: gán `arr[:] = 0`] tensor cũng về 0. Muốn bản sao độc lập, dùng `torch.tensor(arr)`. Với Pandas, lấy `.values` rồi chuyển.

**[6:30 · Slide 9: Bốn phép toán nền tảng]**
Cộng từng phần tử. Nhân vô hướng, ví dụ nhân gradient với learning rate. Hadamard, nhân từng cặp phần tử, ra tensor mới. Và dot product: nhân từng cặp rồi cộng lại, ra đúng một số. Dot là trái tim của mô hình tuyến tính: một neuron chỉ là dot giữa đầu vào và trọng số, cộng bias.

**[7:30 · Slide 10: Broadcasting, hàm toán học, linspace]**
Broadcasting cho phép cộng scalar vào cả tensor mà không cần vòng lặp. Hàm như `sin`, `exp` áp dụng lên từng phần tử; còn `mean`, `max` gộp về một số. Cuối cùng `linspace` tạo các điểm cách đều. Ghép với `sin` rồi vẽ, bạn có ngay đồ thị. Nhớ `.numpy()` trước khi đưa vào matplotlib.
[Chạy code vẽ sin.]

**[8:30 · Slide 11: Tóm tắt]**
Tóm lại: tensor là cấu trúc dữ liệu trung tâm. Kiểm tra `dtype` và `shape` trước khi tính. Cẩn thận chuyện dùng chung bộ nhớ. Bốn phép toán: cộng, nhân vô hướng, Hadamard, dot. Bài sau mình lên hai chiều và gặp đạo hàm tự động. Hẹn bạn ở bài sau, và đừng quên làm lab.
