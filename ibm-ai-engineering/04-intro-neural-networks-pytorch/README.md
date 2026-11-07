# Introduction to Neural Networks and PyTorch

- Thời lượng theo trang Coursera: ~19 giờ, 7 module, 91 mục (20 lab)
- Trạng thái: **đã soạn đủ 7 module** (bạn chưa học). Final Project chỉ có đề + gợi ý, không có lời giải (bài chấm điểm, hạn nộp **Nov 22, 11:59 PM PST**)
- Slug Coursera: `deep-neural-networks-with-pytorch` (id và cách lấy nội dung: skill `coursera-course-fetch`)
- Học sau khóa này: [05 Deep Learning with PyTorch](../05-deep-learning-pytorch/), bắt đầu từ logistic regression ngay chỗ khóa này kết thúc

## Mục tiêu
Từ tensor đến mô hình phân loại đã train xong, toàn bằng PyTorch:
- Làm việc với tensor 1D/2D, đạo hàm tự động (autograd).
- Tạo `Dataset`, dataset ảnh.
- Linear regression và gradient descent (batch, stochastic, mini-batch), vòng lặp train kiểu PyTorch, tách train/validation/test.
- Linear regression nhiều đầu vào/đầu ra.
- Logistic regression, cross-entropy loss, regularization.

Yêu cầu nền (theo Coursera): biết Python cơ bản, quen khái niệm gradient và ma trận.

## Môi trường (uv)
Python do [`uv`](https://docs.astral.sh/uv/) quản lý (`pyproject.toml` + `uv.lock` trong folder này).
```bash
cd ibm-ai-engineering/04-intro-neural-networks-pytorch
uv sync                  # tạo .venv và cài torch, numpy, pandas, matplotlib, jupyterlab
uv run jupyter lab       # mở lab
```
Thêm thư viện: `uv add <gói>`. Lab chạy được trên CPU, không cần GPU.

## Đã soạn
Mỗi bài nằm trong `lessons/<bài>/`: `lesson.md`, `script.md` (kịch bản video), `slides/`, `quiz.md` + `quiz-answers.md` (đáp án riêng), lab `.ipynb` có gợi ý 3 tầng, và `solution/` (chỉ mở khi đã thử xong).

| Bài | Nội dung | Lab |
|---|---|---|
| [m1-l1-tensor-1d](lessons/m1-l1-tensor-1d/lesson.md) | Tensor 1D: dtype, index/slice, view, NumPy/Pandas, phép toán, broadcasting, linspace | `lab.ipynb` (12 bài) |
| [m1-l2-tensor-2d-autograd](lessons/m1-l2-tensor-2d-autograd/lesson.md) | Tensor 2D, nhân ma trận, 3D; autograd, đạo hàm riêng, chain rule | `lab-2d.ipynb` (9), `lab-autograd.ipynb` (9) |
| [m2-l1-datasets](lessons/m2-l1-datasets/lesson.md) | `Dataset`, transform, `Compose`, MNIST/Fashion-MNIST với torchvision | `lab.ipynb` (13 bài, tải MNIST ~65 MB) |
| [m3-l1-linear-regression](lessons/m3-l1-linear-regression/lesson.md) | Hồi quy tuyến tính: forward, `nn.Linear`, `nn.Module`, MSE | `lab.ipynb` (9 bài) |
| [m3-l2-gradient-descent](lessons/m3-l2-gradient-descent/lesson.md) | Gradient descent, learning rate, loss và cost, vòng lặp huấn luyện | `lab.ipynb` (9 bài) |
| [m4-l1-training-workflow](lessons/m4-l1-training-workflow/lesson.md) | SGD, mini-batch, `DataLoader`, `optim`, train/validation/test, lưu mô hình | `lab.ipynb` (12 bài) |
| [m5-l1-multiple-regression](lessons/m5-l1-multiple-regression/lesson.md) | Nhiều đầu vào/đầu ra, shape, `nn.Linear`, xu hướng PyTorch | `lab.ipynb` (12 bài) |
| [m6-l1-logistic-regression](lessons/m6-l1-logistic-regression/lesson.md) | Phân loại, sigmoid, ngưỡng, decision boundary, `Sequential` | `lab.ipynb` (8 bài) |
| [m6-l2-cross-entropy](lessons/m6-l2-cross-entropy/lesson.md) | MLE, cross-entropy, MSE vs BCE với khởi tạo xấu, optimizer nâng cao, regularization | `lab.ipynb` (8 bài) |
| [m7-l1-practice-project](lessons/m7-l1-practice-project/lesson.md) | Mạng nơ-ron phân loại ung thư vú, Iris | `lab.ipynb` (11 bài, tải dữ liệu từ UCI) |
| [m7-l2-final-project](lessons/m7-l2-final-project/brief.md) | Final Project (chấm điểm): đề, 3 tầng gợi ý, ô tự kiểm tra, **không lời giải** | `final-project.ipynb` (8 nhiệm vụ) |

Mọi lời giải (trừ Final Project) đã được chạy thử bằng `uv run`; ô kiểm tra trong bản đề thất bại cho đến khi bạn làm đúng.

**Lưu ý khi học:**
- Lab Cross-Entropy gốc của khóa (Module 6) không có link tải file, nên phần đó viết lại từ video và lab MSE.
- Một số reading dạng widget (podcast tóm tắt, vài reading của Module 3) không đọc được; kiến thức tương ứng lấy từ video.
- Final Exam và các graded quiz do bạn tự làm; ôn bằng `quiz.md` của từng bài.
- Dữ liệu của Final Project gần như nhiễu thuần (độ chính xác ≈ 50%, AUC ≈ 0.5); xem `brief.md` trước khi làm.

## Danh sách bài (mục lục khóa gốc)
Ký hiệu: **V** video, **R** reading/podcast, **L** lab (IBM Skills Network Labs), **PQ** practice quiz, **G** graded quiz/exam, **C** coach (AI). Chỉ có tên bài và checklist; nội dung bài do Claude soạn lại khi bắt đầu từng module.

### Module 1: Exploring Tensors
- Getting Started: Course Introduction (V), Course Overview (R), Helpful Tips (R)
- Lesson 1, Working with One-Dimensional Tensors
  - Introduction to Modern Neural Network (V); Introduction to Matrices and Vectors (V); Introduction to Tensors and Datasets in PyTorch (V)
  - Basics of PyTorch Tensors (C); Understanding 1D Tensors in PyTorch (V); Common Operations in 1D Tensors (V)
  - Lab: Understanding 1D Tensors (L); Practice Quiz: One-Dimensional Tensors (PQ)
- Lesson 2, Working with Two-Dimensional Tensors
  - Introduction to 2D Tensors (V); 2D Tensor Operations (V); Lab: Two-Dimensional Tensors (L)
  - Understanding Differentiation in PyTorch (V); Lab: Differentiation in PyTorch (L); Practice Quiz: Two-Dimensional Tensors (PQ)
- Lesson 3, Module Summary: Podcast: Tensors (R); Graded Quiz: Tensors (G)

### Module 2: Building Datasets in PyTorch
- Creating Simple Datasets (V); Lab: Simple Dataset (L)
- Building Image Datasets (V); Lab: Torch Vision Datasets (L)
- Practice Quiz: Datasets (PQ); Podcast: Datasets (R); Graded Quiz: Datasets (G)

### Module 3: Applying Linear Regression and Gradient Descent
- Lesson 1, Predicting and Training with Linear Regression
  - Linear Regression in PyTorch (V); Lab: Linear Regression 1D: Prediction (L); Linear Regression Prediction (V)
  - Reading: Best Practices for Training Linear Regression Models (R); Training Linear Regression Models (V); Loss Functions (V)
  - Practice Quiz (PQ)
- Lesson 2, Applying Gradient Descent
  - Gradient Descent Basics (V); Reading: Types of Gradient Descent (R); Cost Functions and Batch Gradient Descent (V)
  - Reading: Cost Functions (R); PyTorch Linear Regression Training Slope and Bias (V); Lab: Linear Regression: Prediction (L)
  - Practice Quiz: Gradient Descent (PQ)
- Lesson 3: Podcast (R); Graded Quiz: Linear Regression and Gradient Descent (G)

### Module 4: Training Linear Regression Models the PyTorch Way
- Stochastic Gradient Descent (V); Lab: SGD and Data Loader (L)
- Mini-Batch Gradient Descent (V); Lab: Mini-Batch Gradient Descent (L)
- Optimization in PyTorch (V); Lab: Optimization in PyTorch (L)
- Training, Validation, and Test Split (V), in PyTorch (V); Lab: Train/Validation/Test Split (L)
- Practice Quiz (PQ); Summary and Highlights (R); Graded Quiz: Linear Regression PyTorch Way (G)

### Module 5: Extending Linear Regression to Multiple Inputs and Outputs
- Multiple Linear Regression Training (V); Lab (L)
- Multiple Linear Regression Prediction (V); Lab (L)
- Linear Regression Multiple Outputs (V); Lab (L)
- Multiple Output Linear Regression Training (V); Lab: Training with Multiple Outputs (L)
- Current Trends in PyTorch (V)
- Practice Quiz (PQ); Summary and Highlights (R); Graded Quiz (G)

### Module 6: Applying Logistic Regression for Classification
- Lesson 1, Linear Classifiers and Logistic Regression Predictions
  - Introduction to Linear Classifiers (V); Sigmoid Function and Probability Thresholding (V)
  - Logistic Regression Prediction (V); Lab: Logistic Regression Prediction (L); Practice Quiz (PQ)
- Lesson 2, Optimizing with Cross-Entropy Loss
  - Bernoulli Distribution and Maximum Likelihood Estimation (V); Cross-Entropy Loss in Logistic Regression (V)
  - Applying Cross-Entropy Loss in PyTorch (V); Lab: Logistic Regression Mean Square Error (L); Lab: Logistic Regression Cross Entropy (L)
  - Advanced Optimization and Training Techniques (V); Regularization and Generalization (V); Practice Quiz (PQ)
- Lesson 3: Summary and Highlights (R); Graded Quiz (G)

### Module 7: Final Project, Final Quiz, and Course Wrap-Up
- Practice Project: Neural Network for Breast Cancer Classification (L)
- Final Project Overview, Submission Guidelines (R); Final Project: League of Legends Match Predictor (L)
- Nộp bài: AI graded hoặc peer graded; Final Exam (G)
- Podcast: Course Wrap-up; Congratulations and Next Steps; Team and Acknowledgments

## Tiến độ
- [ ] Module 1: Exploring Tensors (đã soạn bài, chờ bạn học)
- [ ] Module 2: Building Datasets in PyTorch
- [ ] Module 3: Linear Regression and Gradient Descent
- [ ] Module 4: Training Linear Regression the PyTorch Way
- [ ] Module 5: Multiple Inputs and Outputs
- [ ] Module 6: Logistic Regression for Classification
- [ ] Module 7: Final Project và Final Exam
- [ ] Lab đã làm (20 lab)
- [ ] Quiz đã làm
- [ ] Project nhỏ đã đưa vào `../projects/`
