# Quiz · Bài 2: Tensor 2D và autograd

10 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md). Câu có code thì dự đoán trước, chạy sau.

**Câu 1.** `A = torch.tensor([[1, 2, 3], [4, 5, 6]])`. `A.shape`, `A.ndimension()` và `A.numel()` lần lượt là gì?

**Câu 2.** Với cùng `A`, `A[1, 2]` và `A[:, 1]` cho kết quả gì?

**Câu 3.** Với `B = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])`, biểu thức nào lấy được **hai cột đầu của cả ba hàng**?
- A. `B[0:2]`
- B. `B[:, 0:2]`
- C. `B[0:2][0:2]`
- D. `B[2, 0:2]`

**Câu 4.** `X = [[1, 2], [3, 4]]`, `Y = [[1, 0], [0, 1]]` (Y là ma trận đơn vị). Tính `X * Y` và `X @ Y`.

**Câu 5.** `P` có shape `(4, 3)`, `Q` có shape `(3, 5)`, `R` có shape `(4, 5)`. Phép nào hợp lệ, và shape kết quả?
- A. `P @ Q`
- B. `Q @ P`
- C. `P @ R`
- D. `P * Q`

**Câu 6.** Một ảnh màu có shape `(32, 32, 3)` theo dạng (cao, rộng, kênh). Lệnh nào đổi sang dạng `(3, 32, 32)` kiểu PyTorch?

**Câu 7.**
```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)
```
In ra gì? (Gợi ý: đạo hàm của x³.)

**Câu 8.** `u = tensor(2., requires_grad=True)`, `v = tensor(5., requires_grad=True)`, `f = u * v`, rồi `f.backward()`. Giá trị `u.grad` và `v.grad` là bao nhiêu?

**Câu 9.** `x = torch.tensor([1., 2., 3.], requires_grad=True)`, `y = x ** 2`. Gọi `y.backward()` sẽ bị gì? Sửa thế nào để `x.grad` thành `[2., 4., 6.]`?

**Câu 10.** Bạn chạy `backward()` hai lần liên tiếp (mỗi lần tính lại `y`) trên cùng tensor `x` mà không làm gì ở giữa. `x.grad` có bằng gradient của một lần không? Trong huấn luyện thật, ta xử lý việc này thế nào?
