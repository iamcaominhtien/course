# Đáp án · Quiz Bài 1: Tensor 1D

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md). Các kết quả code đã được chạy thử với PyTorch.

| Câu | Đáp án |
|---|---|
| 1 | **B** |
| 2 | **B** |
| 3 | **C** |
| 4 | Shape `(3, 2)`; `view(-1, -1)` báo lỗi |
| 5 | `tensor(99., dtype=torch.float64)` |
| 6 | Tensor số nguyên không có trung bình; đổi sang float |
| 7 | `u * v` = `tensor([4, 10, 18])`; `dot` = `tensor(32)` |
| 8 | `tensor([11, 12, 13])`, broadcasting |
| 9 | `[0.00, 0.25, 0.50, 0.75, 1.00]` |
| 10 | `torch.tensor(arr)` hoặc `.clone()`; matplotlib làm việc với NumPy |

## Giải thích

**Câu 1 (B).** Tensor là mảng n chiều. A sai vì tensor dùng cho mọi loại dữ liệu số. C sai vì tensor chạy được cả CPU lẫn GPU. D sai: tensor 1D chính là vector.

**Câu 2 (B).** List toàn số nguyên cho `int64`, list có số thực cho `float32` (kiểu số thực mặc định của PyTorch).

**Câu 3 (C).** Slice `1:4` lấy index 1, 2, 3 và **không** lấy index 4. Giá trị tương ứng là 20, 30, 40.

**Câu 4.** 6 phần tử, 2 cột nên `-1` được suy ra là 3: shape `(3, 2)`. Với `view(-1, -1)`, PyTorch báo lỗi `only one dimension can be inferred`, vì chỉ được dùng `-1` một lần.

**Câu 5.** `from_numpy` không sao chép dữ liệu, tensor và mảng dùng chung bộ nhớ, nên sửa `arr[0]` thì `t[0]` đổi theo. Kiểu là `float64` vì mảng NumPy mặc định `float64`.

**Câu 6.** Trung bình cần kiểu số thực, mà `[2, 4, 6]` là `int64`. Sửa: `torch.tensor([2, 4, 6]).float().mean()` hoặc tạo tensor từ list số thực. (`std` cũng có lỗi tương tự.)

**Câu 7.** `u * v` là Hadamard, nhân từng cặp phần tử cùng vị trí, kết quả vẫn là **tensor** cùng kích thước. `torch.dot` nhân từng cặp rồi **cộng lại**, kết quả là **một số** (1·4 + 2·5 + 3·6 = 32).

**Câu 8.** Mỗi phần tử cộng thêm 10. Đó là **broadcasting**: scalar được "kéo giãn" cho khớp với kích thước tensor.

**Câu 9.** 5 điểm cách đều từ 0 đến 1, bước 0.25.

**Câu 10.** `torch.tensor(arr)` tạo bản sao độc lập (hoặc `torch.from_numpy(arr).clone()`). Còn `plt.plot` cần dữ liệu kiểu NumPy/list, và tensor cần được đổi qua `.numpy()` (nếu tensor đang theo dõi gradient thì phải `.detach().numpy()`).

## Tự chấm
- 9 đến 10 câu đúng: sang bài 2.
- 6 đến 8 câu đúng: đọc lại các mục liên quan trong `lesson.md`, rồi làm lab.
- Dưới 6 câu: đọc lại `lesson.md` mục 4, 5 trước khi làm lab.
