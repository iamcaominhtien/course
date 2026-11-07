# IBM AI Engineering Professional Certificate (Coursera)

13 khóa con, ~168 giờ, mỗi khóa một folder `NN-...`. Quy tắc mentor chung nằm ở `CLAUDE.md` gốc repo.

## Cấu trúc
- Mỗi bài học nằm trong `NN-<khóa>/lessons/<bài>/`: `lesson.md`, `script.md`, `slides/`, `quiz.md` (+ `quiz-answers.md`), `lab.ipynb`, `solution/`.
- Project nhỏ của từng khóa đưa vào `projects/`.
- Môi trường Python của từng khóa con do `uv` quản lý (xem quy tắc ở `CLAUDE.md` gốc): `pyproject.toml` và `uv.lock` nằm ngay trong folder khóa (ví dụ `04-intro-neural-networks-pytorch/`), chỉ tạo khi khóa bắt đầu dùng Python. Lab của khóa hướng dẫn `uv sync` rồi `uv run jupyter lab`.
- Nguồn nội dung: Claude tự lấy mục lục, transcript video, reading và đề lab (.ipynb) của khóa đang học qua browser pane đã đăng nhập, theo skill `coursera-course-fetch` (`.claude/skills/coursera-course-fetch/`). Bảng id/slug 13 khóa nằm trong `references/ibm-ai-engineer.md` của skill đó.
- Chỉ dùng nội dung gốc làm nguồn: viết lại bằng lời của Claude, không đưa transcript/notebook gốc (bản quyền IBM) vào repo. Không làm graded quiz/final exam hộ.
- Lấy nội dung từng khóa khi người học bắt đầu khóa đó, không lấy trước hàng loạt.
- Cập nhật bảng tiến độ trong `README.md` và checklist trong README của khóa khi hoàn thành.
- `CLAUDE.md` trong folder khóa con chỉ tạo khi khóa đó có quy tắc riêng.
