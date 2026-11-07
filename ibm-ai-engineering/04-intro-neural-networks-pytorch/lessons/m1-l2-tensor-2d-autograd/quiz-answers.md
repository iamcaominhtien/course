# Đáp án · Quiz Bài 2: Tensor 2D và autograd

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md). Các kết quả code đã được chạy thử với PyTorch.

| Câu | Đáp án |
|---|---|
| 1 | `torch.Size([2, 3])`, `2`, `6` |
| 2 | `tensor(6)`; `tensor([2, 5])` |
| 3 | **B** |
| 4 | `X * Y = [[1, 0], [0, 4]]`; `X @ Y = X` |
| 5 | **A**, kết quả `(4, 5)` |
| 6 | `img.permute(2, 0, 1)` |
| 7 | `tensor(27.)` |
| 8 | `u.grad = 5`, `v.grad = 2` |
| 9 | Báo lỗi (kết quả không phải scalar); dùng `y.sum().backward()` |
| 10 | Không, gradient cộng dồn; đặt về 0 mỗi lượt (`x.grad.zero_()`) |

## Giải thích

**Câu 1.** Hai hàng, ba cột: shape `(2, 3)`; số chiều (rank) là 2; tổng số phần tử 2 × 3 = 6.

**Câu 2.** `A[1, 2]`: hàng thứ hai, cột thứ ba, bằng 6. `A[:, 1]`: cột thứ hai của mọi hàng, bằng `[2, 5]`.

**Câu 3 (B).** `B[:, 0:2]` chọn mọi hàng và cột 0, 1. A lấy hai hàng đầu (đủ ba cột). C cắt hàng hai lần liên tiếp. D chỉ lấy một hàng.

**Câu 4.** `X * Y` là Hadamard: từng phần tử nhân với phần tử cùng vị trí, ra `[[1·1, 2·0], [3·0, 4·1]] = [[1, 0], [0, 4]]`. `X @ Y` là nhân ma trận, và nhân với ma trận đơn vị thì giữ nguyên `X`. Đây là ví dụ rõ nhất cho thấy `*` và `@` khác nhau.

**Câu 5 (A).** Quy tắc `(m, n) @ (n, p) → (m, p)`: `(4, 3) @ (3, 5)` ra `(4, 5)`. B sai vì `(3, 5) @ (4, 3)` có 5 ≠ 4. C sai vì `(4, 3) @ (4, 5)` có 3 ≠ 4. D sai vì `*` cần cùng kích thước (hoặc broadcast được) mà `(4, 3)` và `(3, 5)` thì không.

**Câu 6.** `permute(2, 0, 1)` đưa trục kênh (trục 2) lên đầu, rồi đến cao, rộng.

**Câu 7.** y = x³ nên dy/dx = 3x² = 3·9 = 27.

**Câu 8.** f = u·v nên ∂f/∂u = v = 5, ∂f/∂v = u = 2.

**Câu 9.** `y` có 3 phần tử, `backward()` trực tiếp báo lỗi `grad can be implicitly created only for scalar outputs`. Gộp thành scalar bằng `y.sum()` rồi gọi `backward()`. Vì y = Σ xᵢ², ∂/∂xᵢ = 2xᵢ nên `x.grad = [2, 4, 6]`.

**Câu 10.** Không. `.grad` được **cộng dồn**: lần hai cộng thêm vào kết quả lần một, nên gấp đôi (nếu hàm giống nhau). Trong huấn luyện, ta đặt gradient về 0 trước mỗi lượt cập nhật, bằng `x.grad.zero_()` hoặc `optimizer.zero_grad()` (học ở module 4).

## Tự chấm
- 9 đến 10 câu đúng: sang lab, rồi sang module 2.
- 6 đến 8 câu đúng: đọc lại mục 3, 7, 10 của `lesson.md`.
- Dưới 6 câu: đọc lại cả `lesson.md` trước khi làm lab.
