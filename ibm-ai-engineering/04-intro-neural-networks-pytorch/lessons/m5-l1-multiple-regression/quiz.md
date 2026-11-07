# Quiz · Module 5: Nhiều đầu vào, nhiều đầu ra

10 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** Một mô hình hồi quy có 3 đặc trưng đầu vào và 1 đầu ra. Có bao nhiêu trọng số và bao nhiêu bias?

**Câu 2.** `w = [[2.0], [3.0]]`, `b = [[1.0]]`, `x = [[1.0, 2.0]]`. Tính `x @ w + b` và shape của kết quả.

**Câu 3.** Ma trận đầu vào `X` shape `(N, d)`: mỗi hàng và mỗi cột đại diện cho gì?

**Câu 4.** `nn.Linear(4, 3)` nhận `X` shape `(5, 4)`. Kết quả có shape gì? `weight` và `bias` có shape gì?

**Câu 5.** Vì sao `nn.Linear(2, 1).weight` có shape `(1, 2)` chứ không phải `(2, 1)` như `w` trong công thức tay?

**Câu 6.** Bạn có `X` shape `(10, 3)` nhưng tạo nhầm mô hình `nn.Linear(2, 1)`. Chuyện gì xảy ra?

**Câu 7.** Mô hình hồi quy có 2 đặc trưng vào và 1 giá trị ra biểu diễn hình học nào trong không gian 3 chiều (x₁, x₂, y)?

**Câu 8.** Mô hình dự đoán 2 đầu ra từ 3 đặc trưng. Nhãn `y` của một lô 5 mẫu có shape gì? `nn.MSELoss()` tính sai số theo cách nào trên các đầu ra?

**Câu 9.** Dữ liệu có hai cột đặc trưng giống hệt nhau, sinh theo `y = x₁ + x₂ + 1`. Sau huấn luyện, mô hình cho `w = [1.4, 0.65]`, `b ≈ 1`. Mô hình sai ở đâu không? Giải thích.

**Câu 10.** Phân biệt hai cách dùng transfer learning: frozen backbone và fine-tuning.
