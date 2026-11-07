# Quiz · Module 2: Dataset và transform

10 câu, tự làm trước rồi đối chiếu [`quiz-answers.md`](quiz-answers.md).

**Câu 1.** Hai phương thức bắt buộc để một lớp tự viết dùng được như dataset (với `len(ds)` và `ds[i]`) là gì?

**Câu 2.** `ds[3]` gọi phương thức nào của lớp `ds`?
- A. `__call__(3)`
- B. `__len__()`
- C. `__getitem__(3)`
- D. `__init__(3)`

**Câu 3.** Vì sao PyTorch thiết kế dataset trả về từng mẫu theo yêu cầu, thay vì nạp toàn bộ dữ liệu vào một tensor?

**Câu 4.** Mẫu mà dataset trả về thường có dạng gì trong khóa học này?

**Câu 5.** Một transform dạng lớp cần có phương thức nào để gọi được như một hàm (`t(sample)`)?

**Câu 6.** `add_mult()` cộng 1 vào `x` và nhân `y` với 2; `mult()` nhân cả `x` và `y` với 100. Với mẫu `x = [2, 2]`, `y = [1]`, `Compose([add_mult(), mult()])` cho `x` bằng bao nhiêu? Và `Compose([mult(), add_mult()])`?

**Câu 7.** Khi tạo `FashionMNIST(root="./data", train=False, download=True, transform=...)`, `train=False` nghĩa là gì, và `download=True` làm gì?

**Câu 8.** Một ảnh xám 28×28 đã qua `transforms.ToTensor()` có shape gì? Giá trị pixel nằm trong khoảng nào?

**Câu 9.** `transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])` áp lên ảnh PIL 28×28 cho ra tensor có shape gì?

**Câu 10.** Bạn có sẵn một dataset ảnh `base` và muốn thêm một transform lên **ảnh** (không đụng tới nhãn). Cấu trúc lớp bao cần có những phần nào, và `__getitem__` làm gì?
