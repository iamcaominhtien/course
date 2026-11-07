# Module 2 · Bài 1: Dataset và transform trong PyTorch

> Nguồn: Module "Building Datasets in PyTorch" (2 video, 2 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~25 phút. Làm lab: ~1 giờ. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Giải thích vì sao cần lớp `Dataset` thay vì nạp hết dữ liệu vào một tensor.
2. Tự viết một `Dataset` với `__init__`, `__len__`, `__getitem__`.
3. Viết transform dạng lớp có `__call__`, gắn vào dataset, và ghép nhiều transform bằng `Compose` (hiểu thứ tự).
4. Dùng dataset ảnh dựng sẵn của `torchvision` và các transform ảnh (`CenterCrop`, `ToTensor`...).
5. Viết lớp bao (wrapper) để áp transform lên một dataset có sẵn.

## 1. Vì sao cần `Dataset`?
Ở module trước, dữ liệu là những tensor nhỏ nằm trọn trong bộ nhớ. Dữ liệu thật thì khác: hàng chục nghìn ảnh, nằm trên đĩa, mỗi ảnh có nhãn đi kèm, và thường cần xử lý trước khi đưa vào mô hình. Nạp tất cả vào RAM một lần thường không khả thi.

PyTorch giải quyết bằng một quy ước: **dataset là một đối tượng biết hai việc**:
- cho biết nó có bao nhiêu mẫu (`len(dataset)`);
- trả về mẫu thứ `i` khi được hỏi (`dataset[i]`), thường là cặp `(đặc trưng, nhãn)`.

Mẫu nào cần mới đọc và xử lý, nên dữ liệu lớn vẫn chạy được. Quy ước này cũng là thứ `DataLoader` (module 4) dựa vào để chia lô, xáo trộn và nạp song song.

## 2. Tự viết một Dataset
Một lớp con của `torch.utils.data.Dataset` cần ba phần:

```python
import torch
from torch.utils.data import Dataset

class toy_set(Dataset):
    def __init__(self, length=100):                  # 1) khởi tạo dữ liệu
        self.len = length
        self.x = 2 * torch.ones(length, 2)           # mỗi mẫu có 2 đặc trưng, đều bằng 2
        self.y = torch.ones(length, 1)               # nhãn bằng 1

    def __getitem__(self, index):                     # 2) lấy mẫu thứ index
        return self.x[index], self.y[index]

    def __len__(self):                                # 3) số mẫu
        return self.len
```
Dùng thế nào:
```python
ds = toy_set()
len(ds)        # 100
ds[0]          # (tensor([2., 2.]), tensor([1.]))
for x, y in ds:        # lặp được vì có __getitem__
    ...
```
Điểm cần nhớ:
- `ds[i]` gọi `__getitem__(i)`; `len(ds)` gọi `__len__()`. Hai tên này có dấu gạch dưới kép ở hai đầu và phải đúng chính tả.
- Mẫu trả về là một **tuple** `(x, y)`. Quy ước này giữ nguyên suốt khóa học.
- Vì `self.x[idx]` ném `IndexError` khi `idx` ngoài phạm vi, vòng `for` biết khi nào dừng. Muốn tự kiểm tra: `list(ds)` ra đúng `len(ds)` phần tử.
- Dữ liệu mới nên được tải hoặc sinh trong `__init__` (hoặc đọc theo yêu cầu trong `__getitem__` nếu quá lớn).

## 3. Transform: biến đổi mẫu khi lấy ra
Ta thường muốn chuẩn hóa, thêm nhiễu hay đổi kích thước trước khi mô hình dùng. Thay vì sửa dữ liệu gốc, ta viết **transform**: một đối tượng gọi được (callable) nhận một mẫu và trả về mẫu đã biến đổi.

```python
class add_mult:                     # cộng addx vào x, nhân y với muly
    def __init__(self, addx=1, muly=2):
        self.addx, self.muly = addx, muly
    def __call__(self, sample):
        x, y = sample
        return x + self.addx, y * self.muly
```
Áp trực tiếp: `add_mult()(ds[0])` cho `(tensor([3., 3.]), tensor([2.]))`. Tiện hơn, lưu transform trong dataset để mọi lần lấy mẫu đều tự áp dụng: thêm tham số `transform=None` vào `__init__`, lưu `self.transform`, và trong `__getitem__`:

```python
def __getitem__(self, index):
    sample = self.x[index], self.y[index]
    if self.transform:
        sample = self.transform(sample)
    return sample
```
Dùng: `toy_set(transform=add_mult())`. Lợi ích: dữ liệu gốc không đổi, còn cách xử lý thì thay được tùy ý (ví dụ khi huấn luyện dùng thêm nhiễu, khi kiểm thử thì không).

### Ghép nhiều transform: `Compose`
Giả sử có thêm `mult(100)`, nhân cả `x` lẫn `y` với 100.
```python
from torchvision import transforms
pipeline = transforms.Compose([add_mult(), mult()])   # chạy theo thứ tự trong list
```
**Thứ tự quan trọng.** Với mẫu `x = [2, 2]`, `y = [1]`:
- `add_mult` rồi `mult`: `x = (2 + 1) × 100 = 300`, `y = 1 × 2 × 100 = 200`.
- `mult` rồi `add_mult`: `x = 2 × 100 + 1 = 201`, `y = 1 × 100 × 2 = 200`.

Khi kết quả lạ, hãy kiểm tra thứ tự pipeline trước.

## 4. Dataset ảnh dựng sẵn với `torchvision`
`torchvision.datasets` có nhiều dataset ảnh nổi tiếng (MNIST, FashionMNIST, CIFAR...) và `torchvision.transforms` có các phép biến đổi ảnh.

```python
import torchvision.datasets as dsets
import torchvision.transforms as transforms

data = dsets.FashionMNIST(root="./data", train=True, download=True,
                          transform=transforms.ToTensor())
img, label = data[0]       # img: tensor (1, 28, 28), label: số nguyên 0..9
```
Tham số hay gặp:
| Tham số | Ý nghĩa |
|---|---|
| `root` | thư mục lưu dữ liệu |
| `train` | `True` lấy tập huấn luyện, `False` lấy tập kiểm thử |
| `download` | tự tải nếu chưa có |
| `transform` | biến đổi áp lên từng ảnh |

Fashion-MNIST: 60.000 ảnh xám 28×28 của 10 loại quần áo, giày dép. MNIST: 60.000 ảnh chữ số viết tay, cùng kích thước 28×28 (lab dùng bộ này). Hai bộ này hay dùng làm "bài thử" cho mô hình. Lần chạy đầu `download=True` tải khoảng 65 MB về `root`.

Mỗi mẫu cũng là tuple `(ảnh, nhãn)`. Ảnh dạng tensor có thêm một chiều **kênh**: ảnh xám `(1, 28, 28)`, ảnh màu `(3, H, W)`.

### Transform ảnh thường dùng
```python
t = transforms.Compose([transforms.CenterCrop(20),   # cắt giữa 20×20
                        transforms.ToTensor()])      # đổi sang tensor, giá trị về [0, 1]
```
- `ToTensor()` đổi ảnh PIL (hoặc mảng NumPy) sang tensor `float32` và đưa giá trị pixel về [0, 1].
- Có transform chạy trên ảnh PIL, có transform chạy trên tensor; thứ tự trong `Compose` phải phù hợp kiểu dữ liệu ở từng bước.
- Transform có yếu tố ngẫu nhiên (ví dụ `RandomHorizontalFlip(p=0.5)`) hay dùng để tăng dữ liệu (data augmentation) khi huấn luyện.

## 5. Lớp bao (wrapper) cho dataset có sẵn
Đôi khi ta có sẵn một dataset và chỉ muốn thêm bước xử lý ảnh:

```python
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset, self.transform = dataset, transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```
Cấu trúc này giống dataset tự viết, chỉ khác là dữ liệu đến từ dataset khác. Rất hữu ích khi bạn muốn tách phần "đọc dữ liệu" khỏi phần "xử lý".

## Tóm tắt
- Dataset = `__len__` + `__getitem__`; mẫu là tuple `(đặc trưng, nhãn)`.
- Transform là đối tượng gọi được nhận mẫu, trả mẫu mới; gắn vào dataset để tự áp dụng.
- `Compose` chạy các transform theo đúng thứ tự trong list: thứ tự ảnh hưởng kết quả.
- `torchvision` cho sẵn dataset ảnh và transform ảnh; `ToTensor()` thường là bước cuối.
- Module sau dùng dataset này làm nguyên liệu để huấn luyện mô hình hồi quy.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
