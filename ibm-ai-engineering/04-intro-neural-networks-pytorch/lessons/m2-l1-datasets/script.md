# Kịch bản video · Module 2: Dataset và transform

Thời lượng dự kiến: ~9 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Dữ liệu thật không nằm gọn trong một tensor nhỏ. Hôm nay mình học cách PyTorch tổ chức dữ liệu: lớp Dataset và transform.

**[0:30 · Slide 2: Vì sao cần Dataset]** Hàng chục nghìn ảnh trên đĩa, mỗi ảnh một nhãn, cần xử lý trước khi dùng. Nạp hết vào RAM thường không được. PyTorch quy ước: dataset biết hai việc, có bao nhiêu mẫu, và trả mẫu thứ i khi được hỏi. Mẫu nào cần mới đọc.

**[1:30 · Slide 3: Viết một Dataset]** Kế thừa lớp Dataset. Phần một: `__init__` khởi tạo dữ liệu. Phần hai: `__len__` trả số mẫu. Phần ba: `__getitem__` trả mẫu thứ idx dưới dạng tuple x, y.
[Chạy code: tạo toy_set, in len, in ds[0].]
Có `__getitem__` thì vòng for chạy được luôn. Lưu ý chính tả hai dấu gạch dưới ở hai đầu.

**[3:00 · Slide 4: Transform]** Muốn chuẩn hóa hay thêm nhiễu mà không sửa dữ liệu gốc? Viết transform: một lớp có `__call__`, nhận mẫu, trả mẫu mới.
[Chạy code: add_mult áp trực tiếp, rồi lưu vào dataset.]
Gắn vào constructor, mỗi lần lấy mẫu là tự biến đổi.

**[4:15 · Slide 5: Compose và thứ tự]** Compose ghép transform theo thứ tự. Mẫu x bằng 2, 2: add_mult rồi mult ra x bằng 300; mult rồi add_mult ra x bằng 201. Kết quả khác nhau, nên thứ tự quan trọng.
[Chạy code cả hai thứ tự.]

**[5:15 · Slide 6: Dataset ảnh dựng sẵn]** torchvision có Fashion-MNIST và MNIST. Bốn tham số: root, train, download, transform. Mỗi mẫu là tuple ảnh và nhãn; ảnh là tensor có chiều kênh, ví dụ 1 nhân 28 nhân 28.
[Chạy code: tải FashionMNIST, in shape và nhãn.]

**[6:30 · Slide 7: Transform ảnh]** Compose CenterCrop 20 và ToTensor: cắt giữa còn 20 nhân 20, rồi đổi sang tensor, giá trị về khoảng 0 đến 1. Transform ngẫu nhiên như lật ảnh dùng để tăng dữ liệu khi huấn luyện.
[Chạy code, hiện ảnh trước và sau khi cắt.]

**[7:30 · Slide 8: Wrapper]** Có sẵn một dataset và chỉ muốn thêm xử lý ảnh? Viết lớp bao: giữ dataset gốc, trong `__getitem__` lấy ảnh và nhãn rồi áp transform lên ảnh.

**[8:15 · Slide 9: Tóm tắt]** Dataset là len và getitem. Transform là đối tượng gọi được. Compose theo đúng thứ tự. Module sau mình dùng chúng để huấn luyện mô hình đầu tiên. Làm lab nhé.
