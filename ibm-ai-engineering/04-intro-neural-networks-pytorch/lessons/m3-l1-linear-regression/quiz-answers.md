# Đáp án · Quiz Module 3, Bài 1

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | `w`: trọng số (độ dốc); `b`: bias (hệ số chặn) |
| 2 | `[[1.], [3.], [5.]]`, shape `(3, 1)` |
| 3 | Sai số đo và yếu tố ẩn; phân phối Gauss (chuẩn) |
| 4 | `weight` shape `(1, 1)`, `bias` shape `(1,)` |
| 5 | Tham số khởi tạo ngẫu nhiên; cố định seed để kết quả tái lập |
| 6 | **B** |
| 7 | `linear.weight`, `linear.bias` |
| 8 | `(1² + (−2)²) / 2 = 2.5` |
| 9 | Không triệt tiêu dương/âm; phạt nặng sai số lớn |
| 10 | Loss phụ thuộc `w`, `b`; huấn luyện là tìm tham số làm loss nhỏ nhất |

## Giải thích

**Câu 1.** `w` (weight) là độ dốc: `x` tăng 1 đơn vị thì `ŷ` đổi `w` đơn vị. `b` (bias) là hệ số chặn: giá trị `ŷ` khi `x = 0`.

**Câu 2.** 2·1−1 = 1, 2·2−1 = 3, 2·3−1 = 5. Kết quả có cùng số hàng với `x`: `(3, 1)`, mỗi mẫu một dự đoán.

**Câu 3.** Quan sát luôn có sai số đo và chịu ảnh hưởng của yếu tố ta không biết. Nhiễu thường được giả định là Gauss: lệch nhỏ, quanh 0 thì phổ biến; lệch lớn thì hiếm.

**Câu 4.** Một ma trận trọng số `(out_features, in_features) = (1, 1)` và một vector bias `(out_features,) = (1,)`.

**Câu 5.** Tham số của `nn.Linear` được khởi tạo ngẫu nhiên. Đặt seed để mỗi lần chạy ra cùng giá trị khởi tạo (ví dụ seed 1 cho `w ≈ 0.5153`, `b ≈ −0.4414`), giúp kết quả tái lập và so sánh được.

**Câu 6 (B).** `super().__init__()` khởi tạo phần của `nn.Module` (cơ chế theo dõi tham số, lớp con...). Thiếu nó, gán `self.linear = ...` sẽ báo lỗi.

**Câu 7.** `nn.Module` đặt tên tham số theo tên thuộc tính chứa chúng, nên là `linear.weight` và `linear.bias`.

**Câu 8.** Sai số: 2−1 = 1 và 4−6 = −2. Bình phương: 1 và 4. Trung bình: (1 + 4)/2 = **2.5**.

**Câu 9.** (1) Sai số dương và âm không triệt tiêu nhau khi lấy trung bình. (2) Sai số lớn bị phạt nặng hơn nhiều so với sai số nhỏ, nên mô hình ưu tiên giảm các sai lệch lớn.

**Câu 10.** Dự đoán phụ thuộc `w` và `b`, nên loss cũng phụ thuộc chúng: đổi tham số thì loss đổi. Do đó huấn luyện trở thành bài toán tối ưu: tìm giá trị `w`, `b` để loss nhỏ nhất.

## Tự chấm
9 đến 10 đúng: sang bài 2. 6 đến 8: đọc lại mục 2, 4, 6. Dưới 6: đọc lại cả `lesson.md`.
