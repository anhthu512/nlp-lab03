# LAB 03 - Word Representations and Embeddings

Thư mục này bám theo toàn bộ yêu cầu trong W3. Notebook chỉ chứa code và output máy sinh ra; các câu trả lời thuộc nhóm bị cấm dùng AI được làm bằng tay, scan thành PDF và liên kết trong bốn file Markdown tương ứng.

## Cách chạy

1. Đặt `c4-train.00000-of-01024-30K.json.gz` ở thư mục cha của `lab03/`.
2. Cài dependencies: `numpy`, `pandas`, `scipy`, `gensim`, `scikit-learn`, `matplotlib`, `jupyter`, `nbconvert`.
3. Mở terminal tại `lab03/` và chạy `jupyter nbconvert --to notebook --execute word_embedding.ipynb --inplace --ExecutePreprocessor.timeout=1800`.
4. Kiểm tra `results.csv` đã được tạo lại và notebook vẫn giữ toàn bộ output.

Thiết lập chính: 10.000 documents từ C4, seed 42, tokenizer tiếng Anh nhất quán, Word2Vec Skip-gram với negative sampling, `min_count=2`, `epochs=10`. Năm model duy nhất bao phủ ba context window `2/5/10` và ba dimension `50/100/300` mà không train trùng cấu hình.

## Phần cần code và đã có trong notebook

| Mục đề | Nội dung code/output | Nơi lưu |
|---|---|---|
| 10-11 | Tự cài đặt vocabulary, sparse co-occurrence matrix, cosine và `most_similar`; chạy window 1/2/5 | `cooccurrence.py`, notebook, `results.csv` |
| 17-18 | Train Word2Vec; in rõ corpus, vocabulary, dimension, window, min count, epochs, objective; Top-5 cho 7 từ | notebook, `results.csv` |
| 19 | So sánh similarity ở window 2/5/10 | notebook, `results.csv` |
| 20 | So sánh dimension 50/100/300 theo time, size, similarity, analogy, downstream task | notebook, `results.csv` |
| 21 | Chấm điểm và xếp hạng 5 word pairs | notebook, `results.csv` |
| 22 | Chạy hai analogy và lưu Top-5 | notebook, `results.csv` |
| 24 | Semantic search cho query `medical treatment` | notebook, `results.csv` |
| 25 | Sinh candidate và evidence từ corpus để người học tự chọn 3 đúng + 3 sai/bất ngờ | notebook, `results.csv` |

Thứ tự trong notebook bám theo số mục của đề: **10 Experiment 1 -> 11 Core implementation -> 17 Experiment 2 -> 18 Inspect embeddings -> 19 Experiment 3 -> 20 Experiment 4 -> 21-22 Evaluation -> 24 Application -> 25 Error analysis -> 30 lưu deliverables**. Các mục 12-16, 23, 26-29 là câu hỏi lý thuyết/tính tay hoặc quy định nên được chuyển sang các bản scan thay vì tạo Markdown answer cell trong notebook.

## Phần phải tự viết tay, không dùng AI

| Nhóm | Cần dùng output để nhận xét/phân tích? | File chèn bản scan |
|---|---|---|
| Co-occurrence vectors; hai bài cosine; CBOW/Skip-gram training examples; phép tính analogy giả định | **Không**. Làm trước khi chạy/xem output | `calculations.md` |
| Prediction của Bài 3 và Prediction 1-4, đủ Prediction/Reason/Confidence | **Không**. Phải khóa câu trả lời trước calculation/experiment | `prediction.md` |
| So sánh Experiment 1; giải thích Top-5; window; dimension; word-pair ranking; analogy; 3 đúng + 3 sai/bất ngờ | **Có**. Dùng bảng/output và corpus evidence trong notebook/CSV | `error_analysis.md` |
| Ý nghĩa cosine; đối chiếu Bài 3; sparse/dense; ma trận 100k; CBOW vs Skip-gram; analogy tính tay; polysemy; bảng representation; `bank`; individual check | Chủ yếu **không**; chỉ mục đối chiếu prediction tùy chọn dùng output | `reflection.md` |
| Individual learning check | **Không bắt buộc dùng output**; chuẩn bị trả lời miệng cá nhân khoảng 3 phút | Không có file nộp riêng trong đề |

Tên PDF scan dự kiến: `calculations_scan.pdf`, `prediction_scan.pdf`, `error_analysis_scan.pdf`, `reflection_scan.pdf`. Sau khi scan, đặt bốn file cạnh các `.md`; liên kết đã được chuẩn bị sẵn.

## Deliverables

- `README.md`
- `calculations.md`
- `prediction.md`
- `cooccurrence.py`
- `word_embedding.ipynb` (đã chạy, có output)
- `results.csv`
- `error_analysis.md`
- `reflection.md`

Không nộp các model nhị phân vì đề không yêu cầu và notebook có thể train lại từ corpus gốc.

## AI assistance statement

- **Tool:** ChatGPT.
- **Purpose:**kiểm tra implementation, debugging, tối ưu sparse matrix.
- **Generated content:** `cooccurrence.py`, code cells trong `word_embedding.ipynb`, pipeline xuất `results.csv`, cấu trúc các file Markdown.
- **Verification:**  kiểm tra output, CSV và yêu cầu deliverables.

