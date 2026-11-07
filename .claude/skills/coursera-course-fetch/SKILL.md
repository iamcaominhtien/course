---
name: coursera-course-fetch
description: Lấy nội dung một khóa Coursera (mục lục, transcript video, reading, file lab .ipynb) qua browser pane đã đăng nhập, để soạn lesson/quiz/lab trong repo. Dùng khi người học nhờ lấy bài, đọc transcript, lấy mục lục, tải đề lab của một khóa Coursera, hoặc khi chuẩn bị bài cho khóa trong chứng chỉ IBM AI Engineering. Fetch Coursera course outline, lecture transcripts, readings and lab notebooks.
---

# Lấy nội dung khóa Coursera

Mục đích: lấy **nguồn** để Claude soạn lại bài bằng lời của mình (xem `CLAUDE.md` gốc). Đã kiểm chứng trên khóa 1 của chứng chỉ IBM AI Engineering (xem `references/ibm-ai-engineer.md`). Khóa Coursera khác có thể khác cách lấy: làm theo, gặp lỗi thì ghi vào mục "Lỗi thường gặp".

## Ranh giới (đọc trước)
- Người học tự đăng nhập và tự enroll trong browser pane. Claude không nhập mật khẩu, không tạo tài khoản, không enroll hộ.
- Không làm graded quiz, final exam hộ. Practice quiz chỉ đọc khi người học yêu cầu.
- Không bấm **Launch App** của lab trừ khi người học yêu cầu rõ: nó gửi tên và email sang IBM Skills Network và mở tab popup chặn mọi thao tác trình duyệt của Claude. Không cần bấm vì file đề lab tải được trực tiếp (bước 4c).
- Nội dung là bản quyền IBM/Coursera. Chỉ dùng làm nguồn. Không lưu transcript hay notebook gốc vào repo; tải tạm vào scratchpad. Mục lục (tên bài) thì được đưa vào README.

## 0. Chuẩn bị
1. Mở trang tổ chức của người học (chứng chỉ này: `https://www.coursera.org/organizations/olay-facethestemgap`). Thấy "welcome back, <tên>" là đã đăng nhập. Thấy nút Enroll hoặc đăng nhập thì nhờ người học làm trong pane.
2. Làm việc trên một tab thường (`seed`). Cửa sổ đăng nhập mở ra tab `popup-N`: khi nó còn mở, mọi `navigate` đều bị chặn ("Navigation is paused while a page-opened popup..."). Claude không đóng được; nhờ người học bấm X ở tab thứ hai trên thanh tab (không phải X góc trên cùng, nó đóng cả pane). Cookie dùng chung nên tab chính tự có đăng nhập.
3. Trong Bash, `cd` có thể đổi thư mục làm việc của cả session. Dùng đường dẫn tuyệt đối.

## 1. Lấy slug và id
Chạy trong `javascript_tool` trên tab đang ở `coursera.org` (cùng origin, kèm cookie):
```js
// các khóa của một chứng chỉ (specialization), theo đúng thứ tự
const s=await (await fetch('/api/onDemandSpecializations.v1?q=slug&slug=<spec-slug>&fields=courseIds,name',{credentials:'include'})).json();
const ids=s.elements[0].courseIds;
const c=await (await fetch('/api/onDemandCourses.v1?ids='+ids.join(',')+'&fields=slug,name',{credentials:'include'})).json();
c.elements.map(e=>[e.id,e.slug,e.name])
```
Một khóa lẻ: `/api/onDemandCourses.v1?q=slug&slug=<course-slug>&fields=slug,name`. Thứ tự `courseIds` là thứ tự khóa trong chứng chỉ, còn `elements` thì không.

## 2. Lấy mục lục
Đổi `slug` rồi chạy (đã thử: 111 dòng, ~9.5k ký tự cho khóa 1):
```js
const slug='machine-learning-with-python';
const u=`/api/onDemandCourseMaterials.v2/?q=slug&slug=${slug}&includes=modules,lessons,items&fields=moduleIds,onDemandCourseMaterialModules.v1(name,lessonIds),onDemandCourseMaterialLessons.v1(name),onDemandCourseMaterialItems.v2(name,slug,timeCommitment,contentSummary,lessonId,moduleId)&showLockedItems=true`;
const j=await (await fetch(u,{credentials:'include'})).json(), L=j.linked;
const mods=Object.fromEntries(L['onDemandCourseMaterialModules.v1'].map(m=>[m.id,m.name])), les=Object.fromEntries(L['onDemandCourseMaterialLessons.v1'].map(x=>[x.id,x.name]));
let out=[],pm,pl;
for(const i of L['onDemandCourseMaterialItems.v2']){
  if(i.moduleId!==pm){out.push('# '+mods[i.moduleId].trim());pm=i.moduleId;pl=null}
  if(i.lessonId!==pl){out.push('## '+les[i.lessonId].trim());pl=i.lessonId}
  out.push(`- ${i.contentSummary.typeName} ${i.id} ${i.slug} ${Math.round((i.timeCommitment||0)/60000)}m | ${i.name.trim()}`)}
out.join('\n')
```
Mỗi dòng item: `loại id slug phút | tên`. `id` và `slug` dùng để dựng URL ở bước 3.

Dự phòng: connector Coursera. `search_courses` (truyền `partnerName`) lấy `contentId`, rồi `get_course_material(courseId)` trả cùng cấu trúc nhưng chỉ có tên và loại. `search_courses` trả rất nhiều chữ nên chỉ dùng khi API trên hỏng. Đừng dùng `search_videos` (báo "Tool not found").

## 3. Dựng URL từng item
Dùng `https://www.coursera.org/learn/<course-slug>/<loại>/<id>/<item-slug>`. **Không** chèn `/programs/<org>/` (trả trang "Sorry... page does not exist"; dạng đó chỉ đúng cho trang giới thiệu khóa).

| Loại | Đường dẫn | Trạng thái |
|---|---|---|
| `lecture` (video) | `/lecture/<id>/<slug>` | đã kiểm chứng |
| `supplement` (reading, cheat sheet, summary) | `/supplement/<id>/<slug>` | đã kiểm chứng |
| `ungradedLti` (lab, project trên IBM SN Labs) | `/ungradedLti/<id>/<slug>` | đã kiểm chứng |
| `ungradedAssignment` (practice quiz) | chưa thử, đoán `/quiz/<id>/<slug>` | chưa kiểm chứng |
| `staffGraded` (graded quiz, final exam) | chưa thử | chưa kiểm chứng, không làm hộ |
| `coach` | chưa thử | chưa kiểm chứng |
| `ungradedWidget` (podcast, một số reading) | `/ungradedWidget/<id>/<slug>` mở được nhưng **nội dung nằm trong widget, không đọc được** | đã kiểm chứng |
| `gradedLti` (AI chấm), `phasedPeer` (chấm chéo) | `/gradedLti/<id>/<slug>`, `/peer/<id>/<slug>`: đọc được trang hướng dẫn nộp bài (hạn, điểm, Honor Code) | đã kiểm chứng (chỉ đọc) |

## 4. Đọc nội dung
Trang tải chậm: sau `navigate`, `computer wait 4` rồi mới `get_page_text`. Thấy "Loading..." thì chờ thêm. Mỗi lần một trang, nghỉ vài giây giữa các trang, không quét cả khóa một lúc.

**a. Video.** `get_page_text` (max_chars 8000 trở lên) trả luôn transcript dạng `0:00 <lời>`. Không cần bấm tab Transcript, không cần phát video. Mặc định tiếng Anh.

**b. Reading.** `get_page_text` trả đầy đủ bảng và code. Bị cắt ("[truncated]") thì tăng `max_chars`.

**c. Lab.** Trang lab chỉ có mục tiêu và nút Launch App. File đề nằm ở link "HERE":
1. `read_page` với `filter: "interactive"`, tìm link có đuôi `.ipynb` (dạng `https://cf-courses-data.static.labs.skills.network/<hash>/<Tên>.ipynb?t=<số>`). Phần `?t=` đổi theo lần, luôn lấy lại từ trang.
2. Kiểm tra trước: `curl -sSIL "<url>"` xem `content-length` (lab 1: 19.8 KB).
3. Tải vào thư mục tạm riêng: `curl -sSL -o <scratchpad>/<thư-mục-riêng>/<tên>.ipynb "<url>"`.
4. Đọc bằng `python3 -I`, in từng cell. File là notebook đầy đủ (dữ liệu là URL công khai, đáp án nằm trong `<details>`), đủ để viết `lab.ipynb` gợi ý 3 tầng mà không cần mở lab.

## 5. Đưa vào repo
- Theo `ibm-ai-engineering/CLAUDE.md`: mỗi bài ở `NN-<khóa>/lessons/<bài>/` gồm `lesson.md`, `script.md`, `slides/`, `quiz.md`, `lab.ipynb`, `solution/`.
- Mục lục lấy ở bước 2 đưa vào `README.md` của khóa (tên bài và checklist). Không đưa transcript, nội dung reading hay notebook gốc.
- `lesson.md` và `lab.ipynb` viết lại bằng lời của Claude; `lab.ipynb` có gợi ý 3 tầng, đáp án chỉ ở `solution/`.

## Cách nhanh hơn để lấy transcript (đã dùng cho cả 7 module khóa 04)
Không cần mở từng trang video. Trong `javascript_tool` trên tab `coursera.org`, lấy transcript text thuần qua API (cần `courseId` ở dạng không có tiền tố `course~`, ví dụ từ bảng ở `references/`):
```js
window.CID='<courseId>';
window.tx=async(id,name)=>{const j=await (await fetch(`/api/onDemandLectureVideos.v1/${CID}~${id}?includes=video&fields=onDemandVideos.v1(subtitlesTxt)`,{credentials:'include'})).json();
 const u=j.linked['onDemandVideos.v1'][0].subtitlesTxt.en; const t=await (await fetch(u,{credentials:'include'})).text(); return `### ${name} [${id}]\n`+t.replace(/\s+/g,' ').trim();};
(await Promise.all([['<id1>','Tên 1'],['<id2>','Tên 2']].map(([i,n])=>tx(i,n)))).join('\n\n')
```
Một lần gọi lấy được cả module (5 đến 8 video). `subtitlesTxt` còn có nhiều ngôn ngữ khác (`vi`...), nhưng bản `en` là gốc. Link lab `.ipynb` vẫn phải lấy từ trang lab (bước 4c).

Trang nộp bài (`gradedLti`, `peer`) đọc được bằng `document.body.innerText`: có hạn nộp, điểm từng nhiệm vụ, cách nộp, cảnh báo Honor Code. Hãy đọc trước khi chuẩn bị bài chấm điểm.

## Bài chấm điểm (final project, graded quiz, final exam)
- Chỉ soạn **đề, gợi ý, ô tự kiểm tra hình dạng/kiểu dữ liệu**. Không viết lời giải vào repo; muốn kiểm chứng ô kiểm tra thì tạo lời giải riêng ở thư mục tạm, chạy xong để đó.
- Nói rõ cho người học về hạn nộp, điểm và Honor Code (nộp bài do AI làm có thể bị rớt khóa).
- Chạy thử dữ liệu của bài trước khi viết gợi ý: có khi dữ liệu gần như nhiễu (khóa 04: độ chính xác ≈ 50%), cần cảnh báo người học.

## Mẹo khi lấy cả một module (đã dùng cho module 1 khóa 04)
- Gộp nhiều bước vào một `browser_batch`: `navigate` → `computer wait 8` → `get_page_text`, lặp cho 3 video một đợt. Chờ 5 giây đôi khi chưa đủ (đọc ra trang rỗng): dùng 8 giây.
- Link `.ipynb` lấy bằng một dòng `javascript_tool`: `[...document.querySelectorAll('a')].map(a=>a.href).filter(h=>h.includes('.ipynb'))`.
- Nhiều lab đặt tên file notebook chung chung (`1.ipynb`): tải mỗi lab vào một thư mục riêng để khỏi đè nhau. Notebook lớn (hàng trăm cell): đọc bằng script in rút gọn (cắt mỗi cell ~250 ký tự, bỏ cell trang trí) thay vì in nguyên.
- Gần như mọi lab của IBM có ô "Practice" kèm lời giải trong `<details>` hoặc comment HTML. Chỉ dùng để hiểu ý đồ bài; viết đề và lời giải **mới** của mình, rồi chạy lời giải bằng `uv run` trước khi giao (xem quy tắc uv ở `CLAUDE.md` gốc).

## Lỗi thường gặp
| Dấu hiệu | Nguyên nhân và cách xử lý |
|---|---|
| "Sorry... page does not exist" | URL có `/programs/<org>/`, hoặc sai loại/id. Dựng lại theo bước 3. |
| Trang chỉ có "Loading..." | Chờ thêm vài giây rồi đọc lại. |
| "Navigation is paused while a page-opened popup..." | Còn tab `popup-N`. Nhờ người học đóng. |
| Bị chuyển về trang đăng nhập hoặc thấy nút Enroll | Chưa đăng nhập hoặc chưa enroll. Nhờ người học làm. |
| API trả 401/403 hoặc thiếu `linked` | Chưa đăng nhập, hoặc khóa không thuộc tài khoản này. Dùng connector dự phòng. |
| Trang lab không có link `.ipynb` | Lab không phải notebook, hoặc **trang có chữ "Click HERE" nhưng không có thẻ `<a>`** (gặp ở khóa 04, Module 6, lab Cross Entropy: chữ HERE chỉ là text). Không lấy được file; viết lại từ video và lab liền kề, ghi chú trong README khóa, hỏi người học trước khi mở Launch App. |
| `get_page_text` của reading chỉ ra tiêu đề | Reading dạng `ungradedWidget` (podcast, một số reading) nằm trong widget nhúng, không đọc được. Lấy kiến thức từ video tương ứng và ghi chú trong lesson. |
| Dữ liệu tải từ `ucimlrepo`, torchvision | Cần mạng; ghi chú trong lab. `data/` và `*.pt`/`*.pth` đã nằm trong `.gitignore`. |

## Chưa kiểm chứng
Loại `ungradedAssignment`, `staffGraded`, `coach`; khóa Coursera ngoài chứng chỉ này; transcript ngôn ngữ khác; khóa có lab không chạy trên IBM SN Labs. Khi thử xong, cập nhật bảng ở bước 3 và mục này.
