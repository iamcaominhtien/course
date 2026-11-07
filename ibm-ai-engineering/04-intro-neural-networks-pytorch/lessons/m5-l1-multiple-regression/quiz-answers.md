# Đáp án · Quiz Module 5

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | 3 trọng số, 1 bias |
| 2 | `[[9.]]`, shape `(1, 1)` |
| 3 | Hàng: một mẫu; cột: một đặc trưng |
| 4 | Kết quả `(5, 3)`; `weight` `(3, 4)`; `bias` `(3,)` |
| 5 | PyTorch lưu `weight` dạng `(out, in)` và tính `x @ weightᵀ + bias` |
| 6 | Lỗi shape (3 cột ≠ 2 đặc trưng) |
| 7 | Một mặt phẳng |
| 8 | `(5, 2)`; trung bình bình phương sai số trên mọi phần tử |
| 9 | Không sai; chỉ tổng `w₁ + w₂ ≈ 2.05` được xác định |
| 10 | Frozen: giữ cố định backbone, chỉ train lớp đầu ra; fine-tuning: train cả phần backbone với lr nhỏ |

## Giải thích

**Câu 1.** Mỗi đặc trưng một trọng số: 3. Bias luôn một cho mỗi đầu ra: 1.

**Câu 2.** 1·2 + 2·3 + 1 = 9. `x` `(1, 2)` @ `w` `(2, 1)` = `(1, 1)`.

**Câu 3.** Hàng là mẫu (sample), cột là đặc trưng (feature). Khi dự đoán, mỗi hàng của `X` ra một dòng dự đoán.

**Câu 4.** Đầu vào `(5, 4)` qua `Linear(4, 3)` cho `(5, 3)`. `weight` có dạng `(out_features, in_features) = (3, 4)`, `bias` `(out_features,) = (3,)`.

**Câu 5.** Về mặt công thức, ta viết `ŷ = x·W + b` với `W` `(d, m)`. PyTorch chọn lưu ma trận trọng số dạng `(m, d)` (mỗi **hàng** là trọng số của một đầu ra) rồi tính `x @ weight.T + bias`. Kết quả toán học như nhau, chỉ khác cách lưu.

**Câu 6.** `Linear(2, 1)` mong đầu vào có 2 cột, mà `X` có 3 cột nên báo `RuntimeError` về shape không nhân được (`mat1 and mat2 shapes cannot be multiplied`).

**Câu 7.** Hồi quy tuyến tính với 2 đặc trưng: `y = w₁x₁ + w₂x₂ + b` là phương trình mặt phẳng. Huấn luyện là xoay và dịch mặt phẳng cho khớp các điểm.

**Câu 8.** Mỗi mẫu có 2 giá trị đích nên `y` là `(5, 2)`, cùng shape với `ŷ`. `MSELoss` lấy trung bình bình phương sai số trên tất cả `5 × 2 = 10` phần tử, nên cost gộp sai số của mọi đầu ra.

**Câu 9.** Không sai. Vì `x₁ = x₂`, dự đoán là `(w₁ + w₂)·x₁ + b`. Chỉ tổng `w₁ + w₂` ảnh hưởng, nên mô hình tìm ra tổng đúng (≈ 2.05) nhưng cách chia giữa `w₁` và `w₂` phụ thuộc khởi tạo. Bài học: khi đặc trưng trùng lặp hoặc tương quan mạnh, đừng diễn giải từng trọng số.

**Câu 10.** Frozen backbone: lớp đã học sẵn đóng băng, chỉ huấn luyện lớp cuối cho tác vụ mới; nhanh, ít overfit khi dữ liệu ít. Fine-tuning: huấn luyện lớp mới trước, rồi mở khóa một phần backbone và huấn luyện tiếp với learning rate nhỏ để mô hình thích nghi sâu hơn.

## Tự chấm
9 đến 10 đúng: sang lab. 6 đến 8: đọc lại mục 1, 2, 4. Dưới 6: đọc lại cả `lesson.md`.
