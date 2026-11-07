# Đáp án · Quiz Module 4

Chỉ mở sau khi đã làm xong [`quiz.md`](quiz.md).

| Câu | Đáp án |
|---|---|
| 1 | SGD cập nhật sau **mỗi mẫu**; batch GD dùng **cả tập** cho mỗi lần cập nhật |
| 2 | Batch size: số mẫu một lô; iteration: một lần cập nhật; epoch: một lượt hết dữ liệu |
| 3 | 12 iteration/epoch; 120 lần cập nhật |
| 4 | 9 lô; lô cuối 4 mẫu; với `drop_last=True`: 8 lô |
| 5 | SGD; batch GD |
| 6 | Nhỏ: nhiễu hơn, ít bộ nhớ; lớn: mượt hơn, nhiều bộ nhớ |
| 7 | forward, loss, `zero_grad()`, `backward()`, `step()` |
| 8 | `backward()` tính gradient; `step()` cập nhật tham số theo gradient |
| 9 | Để gradient không cộng dồn qua các iteration |
| 10 | Train: học tham số; validation: chọn siêu tham số/mô hình; test: đo cuối, một lần |
| 11 | Train loss giảm nhưng validation loss tăng |
| 12 | Validation loss thấp nhất; `torch.save(model.state_dict(), path)` |

## Giải thích

**Câu 1.** SGD cập nhật tham số sau từng mẫu (nhiều lần cập nhật nhỏ, nhiễu). Batch GD tính gradient trên toàn bộ dữ liệu rồi mới cập nhật một lần (ổn định, nhưng chậm với dữ liệu lớn).

**Câu 3.** 60 / 5 = 12 iteration mỗi epoch. 10 epoch × 12 = 120 lần cập nhật.

**Câu 4.** 60 = 8 × 7 + 4, nên có 8 lô đầy đủ và một lô cuối 4 mẫu: tổng 9 lô. `drop_last=True` bỏ lô cuối: 8 lô.

**Câu 5.** `batch_size=1`: mỗi lần một mẫu ⇒ SGD. `batch_size=len(dataset)`: mỗi lần cả tập ⇒ batch GD.

**Câu 6.** Batch nhỏ: loss dao động nhiều (nhiễu), tốn ít bộ nhớ, cập nhật nhiều lần mỗi epoch. Batch lớn: gradient ổn định, loss mượt, tốn nhiều bộ nhớ, cập nhật ít lần hơn.

**Câu 7.** Thứ tự đúng: `yhat = model(x)`, `loss = criterion(yhat, y)`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. (`zero_grad()` có thể đặt ngay sau `step()` hoặc đầu iteration; điều kiện là xảy ra trước `backward()` kế tiếp.)

**Câu 8.** `backward()` dùng autograd tính gradient của loss theo từng tham số, ghi vào `.grad`. `step()` đọc `.grad` và đổi tham số (với SGD: `tham số -= lr × grad`).

**Câu 9.** `backward()` cộng dồn gradient vào `.grad`. Không xóa thì gradient của iteration trước lẫn vào iteration sau, cập nhật sai.

**Câu 10.** Training để học `w`, `b`. Validation để so sánh các siêu tham số và chọn mô hình trong lúc phát triển (có thể dùng nhiều lần). Test chỉ dùng **một lần** ở cuối để có ước lượng không thiên lệch; dùng nhiều lần để chọn mô hình thì nó trở thành validation.

**Câu 11.** Training loss tiếp tục giảm trong khi validation loss ngừng giảm rồi tăng: mô hình đang học thuộc nhiễu của tập huấn luyện.

**Câu 12.** Chọn theo validation loss, vì nó đo khả năng tổng quát hóa trên dữ liệu chưa thấy. Mô hình có training loss thấp nhất có thể đang overfitting (trong lab, hai mô hình này khác nhau). Lưu bằng `torch.save(model.state_dict(), "best.pt")`.

## Tự chấm
11 đến 12 đúng: sang lab. 8 đến 10: đọc lại mục 2, 4, 5. Dưới 8: đọc lại cả `lesson.md`.
