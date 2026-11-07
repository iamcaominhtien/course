# Quiz · Module 3, Bài 1: Hồi quy tuyến tính, dự đoán và loss

10 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** Trong `ŷ = w·x + b`, `w` và `b` lần lượt là gì?

**Câu 2.** Với `w = 2`, `b = -1` và `x = [[1.0], [2.0], [3.0]]`, `forward(x)` bằng bao nhiêu? Shape của kết quả là gì?

**Câu 3.** Vì sao dữ liệu thật thường có "nhiễu" quanh đường thẳng, và nhiễu thường được giả định theo phân phối nào?

**Câu 4.** `nn.Linear(in_features=1, out_features=1)` tạo ra những tham số nào, shape của chúng là gì?

**Câu 5.** Vì sao khi làm thí nghiệm với `nn.Linear` ta thường gọi `torch.manual_seed(...)` trước khi tạo mô hình?

**Câu 6.** Trong lớp con của `nn.Module`, dòng nào bắt buộc ở đầu `__init__` và vì sao?
- A. `self.forward = None`
- B. `super().__init__()`
- C. `self.parameters = []`
- D. Không dòng nào bắt buộc

**Câu 7.** Với mô hình `model = LR(1, 1)` (lớp ở bài giảng, có thuộc tính `self.linear`), các key của `model.state_dict()` là gì?

**Câu 8.** Tính MSE của `ŷ = [2, 4]` và `y = [1, 6]` bằng tay.

**Câu 9.** Nêu hai lý do MSE dùng **bình phương** sai số thay vì chỉ lấy sai số.

**Câu 10.** "Loss là hàm của tham số" nghĩa là gì, và điều đó cho ta cách nhìn nào về việc huấn luyện?
