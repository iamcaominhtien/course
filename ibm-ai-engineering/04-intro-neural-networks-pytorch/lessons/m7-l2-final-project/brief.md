# Final Project: dự đoán kết quả trận League of Legends

> Đây là **bài chấm điểm** của khóa 04. Claude chỉ soạn **đề, gợi ý và ô tự kiểm tra**; **không có lời giải**, và Claude sẽ không viết bài nộp thay bạn. Lý do: xem mục "Quy tắc trung thực" bên dưới.
> Khung làm bài: [`final-project.ipynb`](final-project.ipynb).

## Thông tin lấy từ trang Coursera (kiểm tra lại trên trang trước khi nộp)
- **Hạn nộp:** Nov 22, 11:59 PM PST. Theo giờ Việt Nam (UTC+7) là khoảng **14:59 ngày 23/11**. Hãy đối chiếu lại trên Coursera.
- **Điểm:** 55 điểm cho 8 nhiệm vụ.
- **Hai cách nộp:**
  1. **AI chấm ("Mark")**, tải notebook lên công cụ nộp bài của IBM Skills Network. Tối đa 60 phút mỗi lần, **không giới hạn số lần nộp**. Bấm Launch App sẽ gửi tên và email của bạn sang IBM Skills Network, nên đó là quyết định của bạn.
  2. **Bạn học chấm chéo (peer-graded):** với mỗi nhiệm vụ, chụp màn hình **ô code cùng kết quả của nó**; code phải chạy không lỗi. Các câu hỏi độc lập: sai một nhiệm vụ vẫn làm được các nhiệm vụ khác.
- Bài thi cuối khóa (**Final Exam**, bài kiểm tra 45 phút chấm điểm) tách riêng. Không có tài liệu nào để Claude chuẩn bị trước, ngoài việc bạn ôn các quiz của Module 1 đến 7.

> Hai trang "Final Project Overview" và "Reading: Submission Guidelines" của khóa là widget nhúng, Claude không đọc được. Bạn nên đọc chúng trực tiếp trên Coursera, đặc biệt phần tiêu chí chấm chi tiết.

## Quy tắc trung thực
Trang nộp bài ghi rõ Coursera Honor Code: bạn xác nhận công việc là của chính bạn, và nộp bài do công cụ AI làm có thể dẫn đến **rớt khóa học hoặc bị khóa tài khoản**. Vì thế, theo đúng vai trò mentor trong `CLAUDE.md`:
- Claude giải thích khái niệm, gợi ý theo 3 tầng, và **review code bạn viết**.
- Claude **không** đưa lời giải đầy đủ, và không viết bài nộp.
- Hãy tự gõ code. Mọi nhiệm vụ đều dùng những thứ bạn đã làm ở các lab trước.

## Điều quan trọng về dữ liệu: đừng mong độ chính xác cao
Claude đã chạy thử toàn bộ quy trình (riêng ở máy, không lưu vào repo) để biết trước bạn sẽ thấy gì:
- Dữ liệu: **1000 trận**, 8 đặc trưng (`kills`, `deaths`, `assists`, `gold_earned`, `cs`, `wards_placed`, `wards_killed`, `damage_dealt`) và nhãn `win` (510 thắng, 490 thua).
- Tương quan của từng đặc trưng với `win` **rất yếu** (cỡ 0.00 đến 0.07). Với cấu hình của đề (SGD, `lr = 0.01`, 1000 epoch), độ chính xác test chỉ khoảng **50% đến 55%** và **AUC khoảng 0.5**: tức là mô hình gần như **không dự đoán tốt hơn tung đồng xu**.
- Đó **không** phải lỗi của bạn, và cũng không phải dấu hiệu code sai. Điều cần làm là: làm đúng quy trình, và **diễn giải đúng**: ở bước 5 và 8, hãy viết rõ rằng AUC ≈ 0.5 nghĩa là mô hình chưa học được tín hiệu dự đoán thật sự, và rằng trọng số đặc trưng nhỏ không có nhiều ý nghĩa khi mô hình không dự đoán được.
- Bài học thực tế: một quy trình chuẩn (chia tập, chuẩn hóa, loss, huấn luyện, đánh giá) cho ra kết quả đáng tin; còn kết quả tốt hay không tùy vào việc dữ liệu có thật sự chứa tín hiệu hay không.

Lưu ý: kết quả cụ thể phụ thuộc seed; con số ở trên là để bạn biết mức đại khái, không phải đáp án.

## Tám nhiệm vụ và điểm
| # | Nhiệm vụ | Điểm | Ý chính |
|---|---|---|---|
| 1 | Tải và tiền xử lý dữ liệu | 10 | `read_csv`, tách `X`/`y`, chia 80/20 (`random_state=42`), `StandardScaler`, tensor |
| 2 | Mô hình logistic regression | 5 | lớp `nn.Module` với `Linear` + sigmoid; `BCELoss`; `SGD(lr=0.01)` |
| 3 | Huấn luyện | 5 | 1000 epoch, in loss mỗi 100 epoch, accuracy train và test |
| 4 | Tối ưu: L2 regularization | 5 | `weight_decay` trong optimizer, huấn luyện lại, đánh giá |
| 5 | Trực quan hóa | 8 | confusion matrix, classification report, đường ROC và AUC, diễn giải |
| 6 | Lưu và nạp mô hình | 8 | `torch.save` / `load_state_dict`, `eval()`, kiểm tra nhất quán |
| 7 | Chọn learning rate | 8 | thử `[0.01, 0.05, 0.1]`, khởi tạo lại mô hình mỗi lần, chọn tốt nhất |
| 8 | Độ quan trọng của đặc trưng | 6 | trọng số `Linear`, DataFrame, sắp xếp theo giá trị tuyệt đối, biểu đồ cột |

Mỗi nhiệm vụ trong notebook có: mô tả bằng lời của Claude, ô code để bạn viết, **3 tầng gợi ý**, và một ô tự kiểm tra (chỉ kiểm tra hình dạng và kiểu dữ liệu, không kiểm tra con số).

## Kế hoạch gợi ý
1. Đọc `brief.md` này và các trang Coursera ghi ở trên.
2. Làm lại một lượt các lab Module 6 và Practice Project nếu cần: gần như mọi thứ bạn cần đã nằm trong đó.
3. Làm lần lượt từng nhiệm vụ trong `final-project.ipynb`. Kẹt thì mở gợi ý theo thứ tự, vẫn kẹt thì hỏi Claude.
4. Chạy lại toàn bộ notebook từ đầu (`Kernel → Restart & Run All`) để chắc code chạy không lỗi, rồi mới nộp.
5. Nếu nộp theo cách chấm chéo, chụp màn hình từng ô code kèm kết quả.

## Nơi chạy notebook
Notebook chạy cục bộ bằng môi trường `uv` của khóa (`uv sync`, `uv run jupyter lab`). Khóa gốc dùng IBM Skills Network Labs. Trước khi nộp bằng công cụ AI chấm, hãy kiểm tra trên trang nộp xem cần tải file notebook từ đâu, vì tôi chưa thử nộp.
