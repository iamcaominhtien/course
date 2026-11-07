# Module 6 · Bài 1: Bộ phân loại tuyến tính và dự đoán bằng logistic regression

> Nguồn: Module "Applying Logistic Regression for Classification", Lesson 1 (3 video, 1 lab). Bài này do Claude soạn lại bằng lời của mình, không chép nguyên văn khóa học.
> Thời gian đọc: ~30 phút. Làm lab: ~1 giờ. Lab: [`lab.ipynb`](lab.ipynb).

## Mục tiêu
Sau bài này bạn có thể:
1. Mô tả bài toán **phân loại** và cách bộ phân loại tuyến tính chia dữ liệu bằng **decision boundary**.
2. Giải thích hàm **sigmoid** biến điểm số tuyến tính thành xác suất, và **ngưỡng** biến xác suất thành nhãn lớp.
3. Dựng mô hình logistic regression bằng `nn.Sequential` và bằng lớp `nn.Module` tự viết.
4. Đọc "độ tự tin" của dự đoán từ xác suất.

## 1. Phân loại và decision boundary
Ở các module trước, đầu ra là một con số liên tục (hồi quy). Với **phân loại**, đầu ra là một **nhãn rời rạc**: thư rác / không rác, bệnh / không bệnh, lớp 0 / lớp 1 / lớp 2.

Dữ liệu được tổ chức như cũ:
- Ma trận đặc trưng `X`: mỗi **hàng** là một mẫu, mỗi **cột** là một đặc trưng.
- Vector nhãn `y`: mỗi phần tử là lớp của mẫu tương ứng (số nguyên).

Bộ phân loại tuyến tính tính một **điểm số** bằng hàm tuyến tính quen thuộc:

> **z = w · x + b**

Nếu dữ liệu chia được thành hai nhóm bằng một đường thẳng, ta nói dữ liệu **tách tuyến tính** (linearly separable), và đường đó là **decision boundary** (ranh giới quyết định): `z > 0` rơi về một phía (một lớp), `z < 0` rơi về phía kia (lớp còn lại).

Dạng của ranh giới theo số chiều đặc trưng:
| Số đặc trưng | Ranh giới |
|---|---|
| 1 | một điểm trên trục số |
| 2 | một đường thẳng |
| 3 | một mặt phẳng |
| nhiều hơn | siêu phẳng (hyperplane) |

Ranh giới nằm đúng tại `z = 0`, tức `w · x + b = 0`. Với một đặc trưng, điểm ranh giới là `x = −b / w`.

### Hàm ngưỡng cứng
Cách đơn giản nhất: `z > 0` thì gán lớp 1, ngược lại lớp 0. Hạn chế: chỉ nhận được nhãn, **không biết mô hình chắc chắn đến đâu**. Một mẫu sát ranh giới và một mẫu cách rất xa đều nhận cùng kiểu nhãn.

## 2. Hàm sigmoid
**Logistic regression** cải thiện bằng cách đưa `z` qua hàm **sigmoid** để ra một **xác suất**:

> **p = σ(z) = 1 / (1 + e^(−z))**

Đầu vào là mọi số thực, đầu ra luôn nằm trong (0, 1), đường cong có hình chữ S:
- `z` rất âm → `p` gần 0.
- `z` rất dương → `p` gần 1.
- `z = 0` → `p = 0.5` (đúng ranh giới).

`p` được hiểu là **xác suất mẫu thuộc lớp 1**.

```python
import torch
z = torch.tensor([-100., 0., 100.])
torch.sigmoid(z)           # tensor([0.0, 0.5, 1.0])
torch.nn.Sigmoid()(z)      # cách khác: là một "lớp" có thể ghép vào nn.Sequential
```

## 3. Từ xác suất đến nhãn: ngưỡng
Sau khi có `p`, dùng một **ngưỡng** (threshold), thường 0.5:
- `p ≥ 0.5` → dự đoán lớp 1;
- `p < 0.5` → dự đoán lớp 0.

(Các tài liệu hay lệch nhau ở trường hợp đúng bằng 0.5; trong thực tế điểm này hiếm khi xảy ra. Hãy chọn một quy ước và dùng nhất quán.) Ngưỡng 0.5 chia không gian đặc trưng thành hai vùng ứng với hai lớp, và nó trùng với `z = 0`, tức ranh giới quyết định.

### Độ tự tin
Ưu điểm lớn của sigmoid là nó cho **xác suất**, nên có thêm thông tin về độ tin cậy:
- `p = 0.95`: rất tự tin lớp 1.
- `p = 0.52`: gần ranh giới, thiếu chắc chắn.
- `p = 0.03`: rất tự tin lớp 0.

Mẫu càng xa ranh giới, `|z|` càng lớn và `p` càng gần 0 hoặc 1.

## 4. Dựng logistic regression trong PyTorch
Logistic regression = **lớp tuyến tính + sigmoid**.

**Cách 1: `nn.Sequential`** (xếp các lớp theo thứ tự):
```python
import torch.nn as nn
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())   # 1 đặc trưng vào
model(torch.tensor([[1.0]]))                           # xác suất, shape (1, 1)
model(torch.tensor([[1.0], [100.0]]))                  # shape (2, 1), mỗi hàng một mẫu
model.state_dict().keys()                              # '0.weight', '0.bias': đánh số theo vị trí lớp
```
Với 2 đặc trưng: `nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())`, đầu vào shape `(N, 2)`.

**Cách 2: lớp `nn.Module` tự viết** (giống Module 3 và 5, thêm sigmoid trong `forward`):
```python
class logistic_regression(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = logistic_regression(2)
model(torch.tensor([[100., -100.], [0., 0.], [-100., 100.]]))
```
Hai cách tương đương về mặt tính toán; khác ở tên tham số (`0.weight` so với `linear.weight`). Cách 2 dễ mở rộng khi mô hình phức tạp hơn.

Có thể ghép nhiều khối: `nn.Sequential(nn.Linear(1, 2), logistic_regression(2))` đi qua một lớp tuyến tính trước rồi mới đến logistic regression. Đây chính là hình dạng của một mạng nơ-ron nhỏ.

### Lưu ý khi dùng
- Mô hình **chưa huấn luyện** có tham số ngẫu nhiên, nên xác suất ra cũng "ngẫu nhiên". Chỉ sau khi huấn luyện, ranh giới mới nằm đúng chỗ.
- Đầu vào lớn (như ±100) làm sigmoid **bão hòa** (gần như đúng 0 hoặc 1). Chi tiết quan trọng khi bàn về gradient ở bài sau.

## Tóm tắt
- Phân loại: nhãn rời rạc; bộ phân loại tuyến tính tính `z = w·x + b`, ranh giới tại `z = 0`.
- Sigmoid đổi `z` thành xác suất trong (0, 1); ngưỡng 0.5 đổi xác suất thành nhãn.
- Logistic regression = `Linear` + `Sigmoid`; dựng bằng `nn.Sequential` hoặc lớp `nn.Module`.
- Xác suất cho biết độ tự tin; sát 0.5 là chưa chắc, gần 0 hoặc 1 là rất chắc.

## Tự kiểm tra
Làm [`quiz.md`](quiz.md), rồi [`lab.ipynb`](lab.ipynb). Lời giải ở [`solution/`](solution/), chỉ mở khi đã thử xong.
