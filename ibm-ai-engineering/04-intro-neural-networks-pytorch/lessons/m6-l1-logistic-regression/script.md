# Kịch bản video · Module 6, Bài 1: Logistic regression, dự đoán

Thời lượng dự kiến: ~10 phút. Mỗi phần ứng với một slide trong [`slides/slides.md`](slides/slides.md).

**[0:00 · Slide 1]** Từ đầu khóa, đầu ra của ta là con số liên tục. Hôm nay đầu ra là một nhãn: spam hay không, bệnh hay không. Đó là phân loại.

**[0:30 · Slide 2: Dữ liệu phân loại]** Đặc trưng nằm trong ma trận X, mỗi hàng một mẫu. Nhãn nằm trong vector y, mỗi phần tử là một lớp: 0, 1, 2. Bộ phân loại tuyến tính tính điểm số z bằng w dot x cộng b.

**[1:30 · Slide 3: Decision boundary]** Nếu một đường chia được hai nhóm, dữ liệu tách tuyến tính. Đường đó là decision boundary tại z bằng 0. Một chiều: một điểm. Hai chiều: một đường. Ba chiều: một mặt phẳng. Nhiều hơn: siêu phẳng.

**[2:45 · Slide 4: Hạn chế của ngưỡng cứng]** Z lớn hơn 0 thì lớp 1, ngược lại lớp 0. Nhận được nhãn, nhưng không biết chắc đến đâu. Mẫu sát ranh giới và mẫu rất xa nhận cùng kiểu nhãn.

**[3:30 · Slide 5: Sigmoid]** Đưa z qua sigmoid: một chia một cộng e mũ trừ z. Đầu vào mọi số thực, đầu ra trong 0 và 1. Z rất âm gần 0, z rất dương gần 1, z bằng 0 ra 0.5. Ta đọc đó là xác suất thuộc lớp 1.
[Chạy code: sigmoid trên tensor trừ 100 đến 100, vẽ đường chữ S.]

**[4:45 · Slide 6: Ngưỡng]** Có xác suất rồi, dùng ngưỡng 0.5 để ra nhãn. Ngưỡng 0.5 trùng với z bằng 0, tức là chính decision boundary. Và xác suất cho biết độ tự tin: 0.52 là lưỡng lự, 0.95 hay 0.03 là rất chắc.

**[5:45 · Slide 7: nn.Sequential]** Logistic regression là lớp Linear cộng Sigmoid. Dùng nn.Sequential xếp hai lớp theo thứ tự. Một đầu vào: Linear 1 1. Hai đầu vào: Linear 2 1. Mỗi hàng của X một xác suất.
[Chạy code: x bằng 1 và X bằng 1, 100; in state_dict thấy tên 0.weight, 0.bias.]

**[7:30 · Slide 8: Lớp tự viết]** Hoặc viết lớp con của nn.Module: self.linear là nn.Linear, forward trả sigmoid của linear. Cùng tính toán với Sequential, chỉ khác tên tham số.
[Chạy code: logistic_regression với 1 rồi 2 đầu vào.]

**[8:45 · Slide 9: Lưu ý]** Mô hình chưa train có tham số ngẫu nhiên nên xác suất cũng ngẫu nhiên. Và đầu vào lớn như 100 làm sigmoid bão hòa, điều này sẽ quan trọng ở bài sau.

**[9:30 · Slide 10: Tóm tắt]** Phân loại, sigmoid, ngưỡng, Sequential hoặc Module. Bài sau: huấn luyện mô hình này đúng cách. Làm lab nhé.
