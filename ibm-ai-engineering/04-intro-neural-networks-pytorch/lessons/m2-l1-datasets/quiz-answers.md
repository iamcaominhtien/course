# Đáp án · Quiz Module 2

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | `__len__` và `__getitem__` |
| 2 | **C** |
| 3 | Dữ liệu lớn không nạp hết được; chỉ đọc/xử lý mẫu khi cần |
| 4 | Tuple `(đặc trưng, nhãn)` |
| 5 | `__call__` |
| 6 | `add_mult` rồi `mult`: `x = [300, 300]`; `mult` rồi `add_mult`: `x = [201, 201]` |
| 7 | Lấy tập kiểm thử; tự tải dữ liệu nếu chưa có |
| 8 | `(1, 28, 28)`; khoảng [0, 1] |
| 9 | `(1, 20, 20)` |
| 10 | Giữ dataset gốc và transform; `__getitem__` lấy `(ảnh, nhãn)`, áp transform lên ảnh, trả `(ảnh, nhãn)` |

## Giải thích

**Câu 1.** `__len__` trả số mẫu để `len(ds)` chạy được. `__getitem__(idx)` trả mẫu thứ `idx` để `ds[idx]` và vòng `for` chạy được. `__init__` thường có nhưng không bắt buộc.

**Câu 2 (C).** Dấu ngoặc vuông gọi `__getitem__`. `__call__` dành cho `obj(...)`, `__len__` cho `len(obj)`.

**Câu 3.** Dữ liệu thật rất lớn và nằm trên đĩa, nên không thể (hoặc không nên) nạp hết vào RAM. Dataset chỉ đọc và xử lý mẫu khi được hỏi, đồng thời cho phép `DataLoader` chia lô, xáo trộn, nạp song song.

**Câu 4.** Tuple `(x, y)`: đặc trưng đứng trước, nhãn (hoặc giá trị mục tiêu) đứng sau.

**Câu 5.** `__call__`. Có nó thì đối tượng dùng như một hàm; `Compose` cũng gọi từng transform theo cách đó.

**Câu 6.** `add_mult` rồi `mult`: (2 + 1) × 100 = **300** cho mỗi phần tử của `x`. `mult` rồi `add_mult`: 2 × 100 + 1 = **201**. Thứ tự khác thì kết quả khác (`y` thì như nhau, 200, vì chỉ có phép nhân).

**Câu 7.** `train=False` lấy tập kiểm thử (test); `train=True` mới là tập huấn luyện. `download=True` tự tải dữ liệu về `root` nếu chưa có.

**Câu 8.** Có thêm chiều kênh: `(1, 28, 28)`, kiểu `float32`, giá trị trong [0, 1].

**Câu 9.** Cắt giữa còn 20×20, rồi `ToTensor` thêm chiều kênh: `(1, 20, 20)`.

**Câu 10.** Lớp con của `Dataset` với `__init__(self, dataset, transform=None)` lưu hai thứ đó; `__len__` trả `len(self.dataset)`; `__getitem__(idx)` lấy `(image, label) = self.dataset[idx]`, nếu có transform thì áp lên `image`, rồi trả `(image, label)`.

## Tự chấm
9 đến 10 đúng: sang lab. 6 đến 8: đọc lại mục 2 và 3. Dưới 6: đọc lại cả `lesson.md`.
