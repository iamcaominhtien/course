# Quiz · Module 6, Bài 1: Logistic regression, dự đoán

10 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** Khác biệt cơ bản giữa bài toán hồi quy và bài toán phân loại về đầu ra là gì?

**Câu 2.** Trong bộ phân loại tuyến tính, decision boundary nằm tại đâu, và nó có dạng gì khi dữ liệu có 2 đặc trưng? 3 đặc trưng?

**Câu 3.** Mô hình một đặc trưng có `w = 2`, `b = -1`. Điểm ranh giới quyết định ở `x` bằng bao nhiêu?

**Câu 4.** Hàm ngưỡng cứng (`z > 0 → 1`) có hạn chế gì so với sigmoid?

**Câu 5.** Tính `sigmoid(0)`. `sigmoid(z)` tiến về giá trị nào khi `z` rất lớn, và khi `z` rất âm?

**Câu 6.** Sigmoid output `p = 0.52` và `p = 0.97`. Mô hình tự tin hơn ở trường hợp nào, vì sao? Mẫu nào gần decision boundary hơn?

**Câu 7.** Với ngưỡng 0.5, `p = 0.8` cho nhãn nào? `p = 0.2`? Nhãn dự đoán này tương ứng với dấu của `z` thế nào?

**Câu 8.** `model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())`. Đưa vào `X` shape `(3, 2)`. Kết quả có shape gì, và các giá trị nằm trong khoảng nào?

**Câu 9.** Các key của `state_dict()` của mô hình ở câu 8 là gì? Nếu viết bằng lớp tự định nghĩa có `self.linear = nn.Linear(2, 1)` thì key là gì?

**Câu 10.** Mô hình mới tạo, chưa huấn luyện, dự đoán xác suất cho nhiều mẫu. Các xác suất đó có ý nghĩa gì không? Vì sao?
