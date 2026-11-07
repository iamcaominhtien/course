---
marp: true
theme: default
paginate: true
title: Dataset và transform trong PyTorch
---

# Dataset và transform
Module 2 · Introduction to Neural Networks and PyTorch

---

# Vì sao cần Dataset?
- Dữ liệu thật: nhiều, nằm trên đĩa, có nhãn, cần xử lý
- Không nạp hết vào RAM
- Quy ước PyTorch: dataset biết
  1. **có bao nhiêu mẫu**: `len(ds)`
  2. **mẫu thứ i**: `ds[i]` → `(x, y)`

---

# Tự viết Dataset
```python
class toy_set(Dataset):
    def __init__(self, length=100):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
```
`ds[0]` → `(tensor([2., 2.]), tensor([1.]))` · `for x, y in ds:` chạy được

---

# Transform: đối tượng gọi được
```python
class add_mult:
    def __init__(self, addx=1, muly=2):
        self.addx, self.muly = addx, muly
    def __call__(self, sample):
        x, y = sample
        return x + self.addx, y * self.muly
```
Lưu trong dataset: mỗi lần lấy mẫu **tự áp dụng**.
Dữ liệu gốc không đổi.

---

# Compose: thứ tự quan trọng
Mẫu `x = [2, 2]`, `y = [1]`

| Pipeline | x | y |
|---|---|---|
| `Compose([add_mult(), mult()])` | (2 + 1) × 100 = **300** | 1 × 2 × 100 = **200** |
| `Compose([mult(), add_mult()])` | 2 × 100 + 1 = **201** | 1 × 100 × 2 = **200** |

---

# Dataset ảnh dựng sẵn
```python
data = dsets.FashionMNIST(root="./data", train=True,
                          download=True,
                          transform=transforms.ToTensor())
img, label = data[0]    # (1, 28, 28), 0..9
```
`root` · `train` · `download` · `transform`
Fashion-MNIST: 60.000 ảnh xám 28×28, 10 loại

---

# Transform ảnh
```python
t = transforms.Compose([transforms.CenterCrop(20),
                        transforms.ToTensor()])
```
- `CenterCrop(20)`: cắt giữa 20×20
- `ToTensor()`: ảnh → tensor float32, giá trị [0, 1]
- Transform ngẫu nhiên (lật ảnh...) → tăng dữ liệu

---

# Lớp bao (wrapper)
```python
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None): ...
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform: image = self.transform(image)
        return image, label
```

---

# Tóm tắt
- Dataset = `__len__` + `__getitem__`, mẫu là `(x, y)`
- Transform = đối tượng có `__call__`
- `Compose`: chạy theo thứ tự, thứ tự ảnh hưởng kết quả
- `torchvision`: dataset ảnh + transform ảnh
- Tiếp theo: dùng dataset để huấn luyện mô hình
