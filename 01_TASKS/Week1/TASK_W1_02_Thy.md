# TASK_02_Thy_Week1 — Dữ liệu mock, huấn luyện ViSoBERT một lần, Core Engine và địa hình sơ bộ

| Mục | Nội dung |
|---|---|
| Vai trò | Nhánh Core: nền móng dữ liệu, hàm ra quyết định, hàm mục tiêu |
| Owner | Thy |
| Reviewers | Ngọc (Leader, nghiệm thu), Vinh (Peer, đối chiếu công thức hàm mục tiêu) |
| Phối hợp | Nga (runner, dùng mock để đo NFE) |
| Mã task gốc | CORE-01 |
| Khung thời gian | Tuần 1 trong kế hoạch 4 tuần, mỗi tuần 7 ngày lịch |
| Deadline tuần | M0b: 23:30 ngày 1 (D+1) · M1: 23:30 ngày 3 (D+3) · Bàn giao địa hình sơ bộ và kết quả Coordinate Ascent ban đầu: 23:30 ngày cuối Tuần 1 |

## 1. Mục tiêu tuần (Weekly Goal)

Tuần 1 nằm trên đường găng của dự án. Mọi thành viên khác phụ thuộc vào `decision.py` và `objective.py`, nên sai sót ở đây nhân lên toàn dự án. Trong tuần này Thy phải:

1. Cung cấp dữ liệu mock đầy đủ cho cả nhóm trong 24 giờ đầu (M0b).
2. Huấn luyện ViSoBERT **đúng một lần** để cache ma trận xác suất thật (M1).
3. Tái lập baseline tại ngưỡng 0.5.
4. Vector hóa hàm ra quyết định và hàm mục tiêu.
5. Tính ngưỡng toàn cục `t_g` và sinh hai file ánh xạ thật (M1).
6. Bàn giao biểu đồ địa hình sơ bộ và kết quả Coordinate Ascent ban đầu cho Ngọc làm slide báo cáo tiến độ cuối Tuần 1.

## 2. Input đầu vào

**Dữ liệu và hợp đồng**
- Bộ ViGoEmotions: 20.664 bình luận, 28 nhãn, chia 8:1:1. Kỳ vọng `N_val = N_test = 2066`, `N_train ≈ 16.532`.
- `src/contract/*` từ Ngọc (M0): `constants.py`, `nfe_tracker.py`, `base_optimizer.py` và các mục 2.1, 2.4, 2.5 của hợp đồng.
- Kịch bản huấn luyện Scenario 1 của ViSoBERT: `lr = 5e-5`, `dropout = 0.2`, `epochs = 12`, hàm mất mát BCEWithLogitsLoss kèm `pos_weight`.

**Tài liệu bắt buộc đọc trước khi viết mã**
- ViGoEmotions (2025): mục chia split 8:1:1, định nghĩa nhãn, Appendix C (quy tắc Neutral).
- Nguyen et al. (EMNLP 2023), ViSoBERT: Scenario 1, tokenization, `pos_weight`.
- Lipton, Elkan & Naryanaswamy (ECML PKDD 2014): tính chất của F1 theo ngưỡng.
- Mục tiêu đọc: hiểu vì sao `pos_weight` kéo xác suất lên cao và gây mất cân bằng (Precision thấp hơn Recall ở ngưỡng 0.5), và nắm vững logic so sánh xác suất của nhãn Neutral.

**Quy tắc cách ly tập Test (Ngọc đã chốt, áp dụng xuyên suốt dự án)**
- Tập Test chỉ được nạp duy nhất tại `src/eval/test_gate.py`, và chỉ ở Tuần 4.
- `data_io.py` tuyệt đối không nạp Test.
- Quy tắc grep trên CI:

```bash
grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py
```

Trong Tuần 1 đến Tuần 3, lệnh này không được trả về kết quả nào. Từ Tuần 4, lệnh này chỉ được trả về `test_gate.py`.

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Xác nhận split với bài báo (làm trước khi huấn luyện)
- Đối chiếu số mẫu train/val/test của bài báo với kỳ vọng 16.532 / 2066 / 2066.
- Nếu lệch, Thy báo Ngọc trước khi tiếp tục, không tự điều chỉnh.

### 3.2 Dữ liệu mock (M0b, hạn 23:30 ngày 1)

Thy viết `02_SAMPLE_DATA/make_mock_data.py` với seed cố định `106`.

| Bộ | N | Mục đích |
|---|---|---|
| `mock_small` | 200 | Unit test chạy dưới 1 giây |
| `mock_fullshape` | 2066 | Đo thời gian và NFE, smoke test thuật toán |

Cách sinh để mock giống dữ liệu thật về hình học:
- **Tần suất dương** mỗi nhãn cảm xúc giảm dần theo cấp số nhân từ khoảng 0.22 (nhóm Head) xuống khoảng 0.045 (nhóm Tail).
- **Neutral** chỉ xuất hiện khi mẫu không có nhãn cảm xúc nào khác. Chỉ số Neutral lấy từ `src/contract/constants.py`.
- **Xác suất:** `P = sigmoid(a·(2Y − 1) + b + ε)` với `ε ~ N(0, σ²)` và `b > 0` để mô phỏng độ lệch do `pos_weight` (ngưỡng tối ưu `t_g > 0.5`). Nhãn Tail dùng `a` nhỏ hơn nhãn Head để F1 thấp hơn.
  - Giá trị khởi điểm gợi ý: `a ∈ [1.0, 2.5]`, `b ≈ 0.8`, `σ ≈ 1.0`. Thy hiệu chỉnh cho đến khi kiểm tra được `t_g > 0.5` và F1 của Tail thấp hơn F1 của Head.
- **Ràng buộc hàng:** mỗi hàng có ít nhất một nhãn bằng 1.
- Không sinh dữ liệu cho mô hình nào khác. Dự án chỉ dùng ViSoBERT.

Ngoài hai bộ dữ liệu, script phải sinh thêm đầy đủ các file sau (Ngọc đã chốt bổ sung):
- Thư mục `mappings/` mock cùng schema với mapping thật.
- `cv_folds_mock.npy` (Vinh cần cho bước 5-fold CV, schema theo EVAL-01 mục B).
- `golden_cases.json` (mục 3.3).
- `meta.json` có sha256 của từng file.

Kiểm tra tái lập: chạy script hai lần, sha256 phải trùng khớp. Kiểm tra shape và dtype theo mục 2.1 của hợp đồng (dự kiến `P` là `float64`, `Y` là `uint8`).

### 3.3 Golden cases (15 nhóm, có đáp án tính tay)

Mỗi ca ghi rõ `P` (tối đa 4 hàng), `theta`, các tham số và kết quả mong đợi. Cột không nêu trong ca gán xác suất 0.01 (dưới mọi ngưỡng).

| Mã | Tình huống | Kết quả mong đợi |
|---|---|---|
| G01 | Neutral 0.70 và một cảm xúc 0.60 cùng vượt ngưỡng | Chỉ giữ Neutral, loại mọi cảm xúc |
| G02 | Như G01 nhưng Neutral 0.55 nhỏ hơn 0.60 | Loại Neutral, giữ cảm xúc |
| G03 | `P[neutral]` đúng bằng xác suất cảm xúc lớn nhất | Loại Neutral |
| G04 | Neutral 0.80 thắng khi có hai cảm xúc cùng vượt ngưỡng (0.60 và 0.55) | Chỉ giữ Neutral (loại cả hai cảm xúc) |
| G05 | Neutral đứng một mình vượt ngưỡng | Giữ Neutral |
| G06 | `exclusive_neutral=False` với đầu vào của G01 | Giữ nguyên `S_i` (cả Neutral và cảm xúc) |
| G07 | Hàng rỗng, fallback `argmax` | Nhãn có `P` lớn nhất |
| G08 | Hàng rỗng, fallback `neutral` | Chỉ Neutral |
| G09 | Hàng rỗng, fallback `head_argmax` | Nhãn có `P` lớn nhất trong tập `head_idx` của `mapping_freq3.json` |
| G10 | Hàng rỗng, fallback `none` | Hàng vẫn rỗng |
| G11 | `P == t` đúng bằng ngưỡng | Dự đoán 1 (so sánh `>=`) |
| G12 | Hòa `argmax` | Chọn chỉ số nhỏ nhất |
| G13 | `theta` sai độ dài | Báo lỗi (loại lỗi theo hợp đồng 2.4, mặc định `ValueError`) |
| G14 | `theta` ngoài `[0.05, 0.95]` hoặc `P` có giá trị lớn hơn 1 | Báo lỗi |
| G15 | Từng `mapping_mode` (`global`, `freq3`, `semantic4`, `per_label`) | Vector 28 chiều đúng; gồm 4 ca con |

Ví dụ cho G01 (mọi ngưỡng bằng 0.5): `P[neutral] = 0.70`, `P[joy] = 0.60`, các cột còn lại 0.01. Khi đó `S = {neutral, joy}`, `m = 0.60`, `0.70 > 0.60`, nên đầu ra chỉ còn Neutral.

### 3.4 Hàm ra quyết định `src/core/decision.py` (M1)

Thy cài `expand_theta(theta, mapping_mode)` và `predict_labels(P, theta, mapping_mode, fallback_strategy, exclusive_neutral)` đúng chữ ký mục 2.4, vector hóa hoàn toàn (không vòng lặp Python theo hàng).

| `mapping_mode` | D |
|---|---|
| `global` | 1 |
| `freq3` | 3 |
| `semantic4` | 4 |
| `per_label` | 28 |

Các bước trong `predict_labels`:
1. Kiểm tra đầu vào: độ dài `theta`, `theta ∈ [0.05, 0.95]`, `P ∈ [0, 1]`.
2. Lọc ngưỡng: `S = (P >= t)` với `t = expand_theta(theta, mapping_mode)` (vector 28 chiều).
3. **Neutral độc quyền** (khi `exclusive_neutral=True`). Với hàng có `neutral ∈ S_i` và `|S_i| > 1`:
   - Tính `m_i = max P[i, c]` trên `c ∈ S_i \ {neutral}`, tức chỉ so với các cảm xúc **cũng đã vượt ngưỡng**.
   - Nếu `P[i, neutral] > m_i` (so sánh chặt): giữ Neutral và loại mọi cảm xúc.
   - Ngược lại (kể cả hòa): loại Neutral và giữ các cảm xúc.
   - Gợi ý vector hóa: dùng `np.where(S_cam_xuc, P, -np.inf).max(axis=1)`.
4. **Fallback cho hàng rỗng**, đủ 4 chế độ:
   - `argmax` (mặc định): nhãn có `P` lớn nhất toàn bộ 28 nhãn.
   - `neutral`: chỉ Neutral.
   - `head_argmax`: `argmax` của `P` trong tập chỉ số `head_idx` đọc từ `mapping_freq3.json`.
   - `none`: để hàng rỗng.
   - Fallback áp dụng sau quy tắc Neutral. Ghi vào docstring rằng nếu Neutral nằm trong `head_idx` thì `head_argmax` có thể chọn Neutral.

**Bộ test (`tests/test_decision.py`):**
- Chạy toàn bộ 15 nhóm golden case.
- **Differential test:** Thy viết `predict_labels_slow` bằng vòng lặp Python từng hàng, không dùng chung mã với bản vector hóa. Với 1000 cặp `(P, theta)` ngẫu nhiên, chạy mọi tổ hợp `mapping_mode × fallback_strategy × exclusive_neutral` (4 × 4 × 2 = 32 tổ hợp). Kết quả hai bản phải **bằng nhau tuyệt đối**.
  - Để kích hoạt ca biên, lượng tử hóa một phần `P` và `theta` theo bội của 0.05 nhằm sinh các trường hợp hòa và `P == t`.
- **Tính bất biến:** gọi hai lần cùng đầu vào trả cùng kết quả; `P` không bị sửa (so sánh với bản sao).

### 3.5 Huấn luyện ViSoBERT một lần và cache (M1, hạn 23:30 ngày 3)

Thy viết `src/training/train_visobert.py` theo Scenario 1. **Không huấn luyện lại lần hai.** Script chặn việc chạy lại: nếu `meta.json` đã ghi checkpoint thì script từ chối chạy.

- Công thức: `pos_weight_c = (N_train − n_c) / n_c`. Nếu bài báo nêu công thức khác thì theo bài báo. Ghi rõ lựa chọn trong `train_config.json`.
- **Chọn checkpoint (Ngọc đã chốt):** chọn epoch có Val Macro-F1 (tại ngưỡng 0.5) cao nhất. Ghi `ckpt_selection = "best_val"` và epoch được chọn vào `train_config.json`.
- Trong lúc huấn luyện, mỗi epoch chỉ đánh giá trên **Val**. Suy luận trên Test chỉ thực hiện một lần, sau khi checkpoint đã được chọn.
- `train_config.json` ghi đủ: siêu tham số, seed, công thức `pos_weight`, `n_c` từng nhãn, `N_train`, `ckpt_selection`, `selected_epoch`.
- Xuất `P_val.npy`, `P_test.npy` bằng `torch.sigmoid(logits)` ở chế độ `eval()`, `float64`, giữ nguyên thứ tự hàng của split. Xuất `Y_val.npy`, `Y_test.npy` (`uint8`).
  - Vòng xuất duyệt qua danh sách split và dựng tên file động (`f"P_{split}.npy"`). Không để các chuỗi bị grep chặn trong mã.
  - Macro-F1, Macro-Precision, Macro-Recall tại ngưỡng 0.5 của Test chỉ tính từ mảng trong bộ nhớ ngay tại bước xuất này (một lần duy nhất) để ghi vào báo cáo tái lập. Thy không nạp lại Test từ đĩa ở bất kỳ mã nào khác, và không tính F1 từng nhãn trên Test trong tuần này.
- Lưu checkpoint ra ngoài git (Drive hoặc HF private). Ghi đường dẫn và sha256 của checkpoint cùng các file `.npy` vào `meta.json`.
- Kiểm tra `P_val` và `P_test`: shape `(2066, 28)`, `float64`, trong `[0, 1]`, không NaN.

### 3.6 `src/core/data_io.py`
- Chỉ cung cấp hàm nạp **Val** (`load_split("val")`) và hàm đọc thống kê train từ `train_config.json`.
- Nếu được gọi với split khác `"val"` thì báo lỗi. Thy viết `tests/test_data_io.py` kiểm tra hành vi này.
- Trong mã không xuất hiện các chuỗi `P_test`, `Y_test` hay `_test.npy`.

### 3.7 Tái lập baseline tại ngưỡng 0.5 (M1)

- Tính Macro-F1, Macro-Precision, Macro-Recall tại ngưỡng 0.5 trên Val (từ `data_io`) và Test (từ bước xuất ở mục 3.5).
- Mốc: Val Macro-F1 khoảng 62.33%, Test Macro-F1 khoảng 61.50%, dung sai ±0.5 điểm phần trăm.
- Kiểm tra tính chất mất cân bằng: Macro-Precision (59.08%) thấp hơn rõ so với Macro-Recall (64.40%). File kế hoạch không nêu hai số này thuộc split nào, nên Thy hỏi Ngọc rồi báo cáo kèm sai lệch cho từng split.
- Đối chiếu `macro_f1` bản NumPy với `sklearn.metrics.f1_score(average="macro", zero_division=0)` trên Val: sai số tuyệt đối ≤ `1e-9`.
- Ghi vào `baseline_at_0.5` trong `meta.json` và viết `reports/baseline_reproduction.md`. Báo cáo này là nguồn số liệu Test Macro-F1@0.5 cho các tuần sau.

### 3.8 Hàm mục tiêu `src/core/objective.py` và ngưỡng `t_g` (M1)

Thy cài `per_label_f1`, `macro_f1`, `make_objective(P, Y, mapping_mode, fallback_strategy, exclusive_neutral, lam, t_g)` đúng mục 2.5:

- `F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c)`, bằng 0 khi mẫu số bằng 0 (khớp `zero_division=0`).
- `J_λ(θ) = MacroF1(Y, Ŷ(P, θ)) − λ · (1/D) · ‖θ − t_g·1‖₂²`

**Yêu cầu tốc độ:** trung vị ≤ 3 ms trên `mock_fullshape` và trên ma trận Val thật `(2066, 28)`, 1 luồng CPU, đo 1000 lần.
- Chuyển `Y` sang `bool` một lần, ngoài vòng lặp nóng.
- Đếm TP, FP, FN bằng `np.einsum` hoặc phép nhân ma trận, tránh cấp phát lại.
- Không gọi `sklearn` trong hàm nóng.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/time_objective.py | tee 03_SUBMISSIONS/Thy/test_logs/timing_objective.txt
```

```python
rng = np.random.default_rng(106)
thetas = rng.uniform(0.05, 0.95, size=(1000, 28))
times_ms = []
for th in thetas:
    t0 = time.perf_counter()
    f(th)
    times_ms.append((time.perf_counter() - t0) * 1e3)
print("median_ms =", np.median(times_ms))
```

**Ngưỡng toàn cục `t_g` (LD-9):** grid search 1 chiều trên Val, miền `[0.05, 0.95]`, bước 0.001 (901 điểm), `λ = 0`. Lưu `artifacts/t_g.json` và đường cong `J(t)` vào `artifacts/j_curve_1d.npy`.

**Test (`tests/test_objective.py`):** `macro_f1` khớp sklearn ở `1e-9`; `make_objective` đúng trên các ca tính tay; tốc độ trung vị ≤ 3 ms.

### 3.9 File ánh xạ thật (hoàn thành trước hạn M1)
- `configs/mappings/mapping_freq3.json`: gồm các khóa `head_idx` (9 chỉ số), `mid_idx` (10 chỉ số), `tail_idx` (9 chỉ số), xếp theo số mẫu dương `n_c` ở tập huấn luyện, kèm tên nhãn và `n_c`. Khóa `head_idx` là nguồn của fallback `head_argmax`.
- `configs/mappings/mapping_semantic4.json`: kèm bảng đối chiếu nhãn ViGoEmotions sang nhóm.
- Ngọc review rồi đóng băng sha256 cho cả hai file.

### 3.10 Địa hình sơ bộ và Coordinate Ascent ban đầu (sau M1, đến hết Tuần 1)
- Chỉ dùng Val, `λ = 0`.
- Vẽ `outputs/figures/landscape_1d_global.png` (đường cong `J(t)`, đánh dấu `t_g` và 0.5).
- Vẽ `outputs/figures/landscape_2d_head_tail.png` (lát cắt `(t_head, t_tail)` với `t_mid = t_g`).
- Cài phiên bản đầu của `coordinate_ascent.py` (kế thừa `BaseOptimizer`, chỉ `_search`; đặc tả đầy đủ ở Tuần 2). Chạy thử D = 1 và D = 3 với NFE = 1000: trước hết trên `mock_fullshape` để kiểm tra tính đúng đắn, sau đó trên Val thật.
- Dùng 5 seed riêng, không trùng 0–19, 100–104, 200–399 (đề xuất 500–504).
- Bàn giao cho Ngọc: hai biểu đồ, bảng `J` tốt nhất, NFE đã dùng, mức cải thiện so với ngưỡng 0.5 và so với `t_g`.

### 3.11 Xử lý sự cố
- **R-1 (baseline lệch quá ±0.5%):** kiểm tra công thức `pos_weight`, cách chọn checkpoint và tiền xử lý văn bản tiếng Việt. Nếu mô hình hội tụ ở điểm lân cận, báo ngay Ngọc để ghi biên bản xác nhận.
- **R-2 (hàm mục tiêu vượt 3 ms):** nguyên nhân thường là vòng lặp theo mẫu hoặc gọi sklearn trong hàm nóng. Chuyển `Y` sang `bool` một lần, tính TP, FP, FN bằng `einsum` hoặc phép nhân ma trận.

## 4. Output tuần

**Mã nguồn:** `02_SAMPLE_DATA/make_mock_data.py`, `src/core/data_io.py`, `src/core/decision.py`, `src/core/objective.py`, `src/training/train_visobert.py`, bản đầu `src/optimizers/local/coordinate_ascent.py`.

**Dữ liệu và artifact:**
- `02_SAMPLE_DATA/mock_small/`, `mock_fullshape/`, mock mappings, `cv_folds_mock.npy`, `golden_cases.json`.
- `src/data/P_val.npy`, `Y_val.npy`, `P_test.npy`, `Y_test.npy`, `meta.json`, `train_config.json`.
- `configs/mappings/mapping_freq3.json`, `mapping_semantic4.json`.
- `artifacts/t_g.json`, `artifacts/j_curve_1d.npy`.

**Test:** `tests/test_decision.py`, `tests/test_objective.py`, `tests/test_data_io.py`, và các log `timing_objective.txt`, `golden_test_output.txt`, `differential_test_1000.txt` trong `03_SUBMISSIONS/Thy/test_logs/`.

**Biểu đồ:** `landscape_1d_global.png`, `landscape_2d_head_tail.png`.

**Báo cáo:** `reports/baseline_reproduction.md`.

## 5. Definition of Done (DoD Tuần 1)

- [ ] Split đã xác nhận với bài báo (hoặc đã báo Ngọc nếu lệch).
- [ ] `make_mock_data.py` chạy hai lần cho sha256 trùng khớp; shape và dtype khớp mục 2.1.
- [ ] Đã giao đúng hạn M0b: `mock_small`, `mock_fullshape`, mock mappings, `cv_folds_mock.npy`, `golden_cases.json`.
- [ ] Đã huấn luyện đúng một lần; `ckpt_selection = "best_val"` ghi trong `train_config.json`; checkpoint ngoài git, đường dẫn trong `meta.json`.
- [ ] `P_val`, `P_test` đúng shape `(2066, 28)`, `float64`, trong `[0, 1]`, không NaN.
- [ ] Baseline 0.5 trong khoảng 62.33% ± 0.5% (Val) và 61.50% ± 0.5% (Test), hoặc có giải thích bằng văn bản được Ngọc ký.
- [ ] `macro_f1` khớp sklearn ở `1e-9`.
- [ ] `predict_labels` qua 15 golden case; differential test 1000 cặp × 32 tổ hợp bằng nhau tuyệt đối; test bất biến xanh.
- [ ] `make_objective` trung vị ≤ 3 ms (log đo kèm).
- [ ] `t_g.json` và `j_curve_1d.npy` đã lưu; `mapping_freq3.json` (có `head_idx`) và `mapping_semantic4.json` hoàn thành trước M1, đã được Ngọc đóng băng sha256.
- [ ] `data_io.py` chỉ nạp Val và từ chối split khác (có test).
- [ ] `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` không trả về kết quả nào.
- [ ] Biểu đồ 1D và 2D cùng kết quả Coordinate Ascent D = 1, 3 đã bàn giao cho Ngọc đúng hạn cuối Tuần 1.