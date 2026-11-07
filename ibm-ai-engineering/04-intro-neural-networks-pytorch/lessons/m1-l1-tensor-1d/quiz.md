# Quiz · Bài 1: Tensor 1D

10 câu, tự làm không xem tài liệu, rồi mới đối chiếu [`quiz-answers.md`](quiz-answers.md). Câu có code thì hãy **dự đoán trước**, sau đó mới chạy để kiểm tra.

**Câu 1.** Phát biểu nào đúng nhất về tensor trong PyTorch?
- A. Tensor là kiểu dữ liệu chỉ dùng cho ảnh.
- B. Tensor là mảng nhiều chiều; scalar, vector, matrix là các trường hợp riêng của nó.
- C. Tensor chỉ tồn tại trên GPU.
- D. Tensor 1D và vector là hai thứ khác nhau hoàn toàn.

**Câu 2.** `torch.tensor([1, 2, 3]).dtype` và `torch.tensor([1.0, 2.0, 3.0]).dtype` lần lượt là gì?
- A. `float32`, `float32`
- B. `int64`, `float32`
- C. `int32`, `float64`
- D. `int64`, `int64`

**Câu 3.** Với `t = torch.tensor([10, 20, 30, 40, 50])`, kết quả của `t[1:4]` là:
- A. `tensor([10, 20, 30, 40])`
- B. `tensor([20, 30, 40, 50])`
- C. `tensor([20, 30, 40])`
- D. `tensor([20, 30])`

**Câu 4.** `x = torch.tensor([1., 2., 3., 4., 5., 6.])`. Hình dạng của `x.view(-1, 2)` là gì? Và `x.view(-1, -1)` sẽ xảy ra điều gì?

**Câu 5.**
```python
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
arr[0] = 99.0
print(t[0])
```
In ra gì? Giải thích ngắn.

**Câu 6.** Vì sao `torch.tensor([2, 4, 6]).mean()` báo lỗi, và cách sửa gọn nhất là gì?

**Câu 7.** Cho `u = torch.tensor([1, 2, 3])` và `v = torch.tensor([4, 5, 6])`. Tính `u * v` và `torch.dot(u, v)`. Hai kết quả khác nhau ở điểm cốt lõi nào?

**Câu 8.** `torch.tensor([1, 2, 3]) + 10` cho kết quả gì, và hiện tượng này tên là gì?

**Câu 9.** `torch.linspace(0, 1, steps=5)` trả về những giá trị nào?

**Câu 10.** Bạn muốn một tensor lấy dữ liệu từ mảng NumPy nhưng **không bị ảnh hưởng** khi mảng NumPy đổi sau này. Dùng cách nào? Và vì sao cần gọi `.numpy()` trước khi đưa tensor vào `plt.plot`?
