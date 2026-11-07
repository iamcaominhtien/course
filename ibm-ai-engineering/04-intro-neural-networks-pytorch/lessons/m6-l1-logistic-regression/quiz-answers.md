# Đáp án · Quiz Module 6, Bài 1

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | Hồi quy: đầu ra liên tục; phân loại: nhãn rời rạc |
| 2 | Tại `z = 0`; 2 đặc trưng: đường thẳng; 3 đặc trưng: mặt phẳng |
| 3 | `x = 0.5` |
| 4 | Chỉ cho nhãn, không cho biết độ tự tin |
| 5 | `0.5`; tiến về 1; tiến về 0 |
| 6 | `0.97` tự tin hơn; `0.52` gần ranh giới hơn |
| 7 | Lớp 1; lớp 0; `p ≥ 0.5` ⇔ `z ≥ 0` |
| 8 | `(3, 1)`; trong (0, 1) |
| 9 | `0.weight`, `0.bias`; `linear.weight`, `linear.bias` |
| 10 | Không có ý nghĩa; tham số còn ngẫu nhiên |

## Giải thích

**Câu 1.** Hồi quy dự đoán một số thực (giá nhà, nhiệt độ). Phân loại dự đoán một lớp trong tập hữu hạn (0/1, hoặc nhiều lớp).

**Câu 2.** Ranh giới là tập điểm có `z = w·x + b = 0`. 1 đặc trưng: một điểm; 2: đường thẳng; 3: mặt phẳng; nhiều hơn: siêu phẳng.

**Câu 3.** `2x − 1 = 0` ⇒ `x = −b/w = 0.5`. Với `x > 0.5` thì `z > 0` và mô hình nghiêng về lớp 1.

**Câu 4.** Ngưỡng cứng chỉ cho nhãn 0/1. Mẫu sát ranh giới và mẫu cách rất xa đều nhận một nhãn như nhau. Sigmoid cho xác suất liên tục nên biết độ tin cậy.

**Câu 5.** `1/(1 + e⁰) = 0.5`. `z → +∞` thì `p → 1`; `z → −∞` thì `p → 0`.

**Câu 6.** `0.97` sát 1 nên mô hình rất tự tin lớp 1. `0.52` gần 0.5 nên sát ranh giới, thiếu chắc chắn.

**Câu 7.** `0.8 ≥ 0.5` ⇒ lớp 1; `0.2 < 0.5` ⇒ lớp 0. Vì `σ(z) ≥ 0.5` đúng khi `z ≥ 0`, ngưỡng 0.5 chính là ranh giới `z = 0`.

**Câu 8.** Mỗi hàng của `X` (một mẫu) cho một xác suất: kết quả `(3, 1)`. Sigmoid luôn cho giá trị trong (0, 1).

**Câu 9.** `nn.Sequential` đánh số lớp theo vị trí: lớp Linear là lớp 0 nên key là `0.weight` và `0.bias`. Lớp tự viết dùng tên thuộc tính: `linear.weight`, `linear.bias`.

**Câu 10.** Không. Tham số được khởi tạo ngẫu nhiên nên ranh giới nằm tùy tiện; xác suất chỉ có ý nghĩa sau khi huấn luyện.

## Tự chấm
9 đến 10 đúng: sang bài 2. 6 đến 8: đọc lại mục 1 đến 3. Dưới 6: đọc lại cả `lesson.md`.
