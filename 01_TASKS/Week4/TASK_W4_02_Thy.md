# TASK_02_Thy_Week4 — Hoàn thiện Bảng 4, kiểm chứng dịch chuyển ngưỡng và nội dung báo cáo

| Mục | Nội dung |
|---|---|
| Vai trò | Nhánh Core: phân tích từng nhãn, viết báo cáo, chuẩn bị bảo vệ |
| Owner | Thy |
| Reviewers | Ngọc (Leader, biên tập tổng thể và nghiệm thu Bảng 4), Vinh (Peer, cung cấp θ*) |
| Mã task gốc | CORE-01 |
| Khung thời gian | Tuần 4 trong kế hoạch 4 tuần, mỗi tuần 7 ngày lịch |
| Deadline tuần | 23:30 ngày cuối Tuần 4. Mốc M6 (bảo vệ) chưa có ngày cụ thể trong kế hoạch nên Thy hỏi Ngọc để chuẩn bị slide đúng lịch. |

## 1. Mục tiêu tuần (Weekly Goal)

1. Nhận vector ngưỡng tối ưu θ* (D = 28, λ = λ*) từ Vinh sau khi Cổng Test mở.
2. Hoàn thiện 100% Bảng 4.
3. Kiểm chứng thực nghiệm hành vi dịch chuyển ngưỡng giữa nhóm Head và nhóm Tail.
4. Soạn nội dung nhánh Core cho báo cáo toàn văn.
5. Xuất biểu đồ độ phân giải cao và chuẩn bị phần trình bày bảo vệ.

## 2. Input đầu vào

- θ* (D = 28, λ = λ*) do Vinh bàn giao sau khi Ngọc mở Cổng Test. θ* phải được tìm **chỉ trên Val** (và CV). Test chỉ dùng để đánh giá cuối cùng, không dùng để chỉnh θ*.
- `P_val.npy`, `Y_val.npy` (qua `data_io`), `outputs/tables/per_label_table4.csv` từ Tuần 3, `landscape_summary.csv`, `reports/baseline_reproduction.md`.
- Cấu hình quyết định đúng như lúc chạy benchmark (mapping `per_label`, fallback, `exclusive_neutral`) do Vinh cung cấp.
- Khung báo cáo toàn văn của Ngọc.
- Tài liệu bắt buộc đọc:
  - Pillai, Fumera & Roli (Pattern Recognition 2013), *Threshold Optimisation for Multi-label Classifiers*.
  - Demszky et al. (ACL 2020), *GoEmotions*.
  - Mục tiêu: phân tích hiện tượng phân hóa ngưỡng giữa nhóm phổ biến (Head) và nhóm hiếm (Tail).

**Quy tắc cách ly Test.** Tập Test chỉ được nạp duy nhất tại `src/eval/test_gate.py`, và chỉ từ Tuần 4. Lệnh sau chỉ được trả về duy nhất `test_gate.py`:

```bash
grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py
```

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Tiếp nhận θ*
- Kiểm tra: shape `(28,)`, giá trị trong `[0.05, 0.95]`, không NaN.
- Ghi sha256 của file θ* và nguồn gốc (lượt chạy nào, seed nào, λ* bằng bao nhiêu) vào `reports/per_label_findings.md`.
- Xác nhận với Vinh rằng θ* không dùng thông tin Test.

### 3.2 Hoàn thiện `outputs/tables/per_label_table4.csv`

**Phần Val** (script `scripts/fill_table4.py`, đặt ngoài `src/`, chỉ nạp Val qua `data_io`):
- `t_star` bằng `θ*_c`.
- `f1_opt_val` là F1 của nhãn `c` khi dự đoán bằng `predict_labels(P_val, θ*, "per_label", ...)`.
- `delta_f1_val = f1_opt_val − f1_base_val` (đơn vị điểm phần trăm).

**Phần Test** (sử dụng cổng `load_test()`, Thy tuyệt đối **không chỉnh sửa** mã nguồn `src/eval/test_gate.py` của Vinh):
- Thy viết một script đánh giá độc lập đặt tại `scripts/eval_test_per_label.py` (hoặc tích hợp an toàn vào `scripts/fill_table4.py` với cờ --eval-test, chỉ chạy sau khi Ngọc cấp LOCK_FREEZE.json).
- Trong script này, Thy gọi hàm load_test() từ module của Vinh để nhận ma trận P_test, Y_test:

``` bash
from src.eval.test_gate import load_test
P_test, Y_test = load_test()
```

- Sử dụng hàm ra quyết định sẵn có của Thy predict_labels và hàm tính per_label_f1 trên tập Test nhận được cho 2 cấu hình:
   1. Ngưỡng baseline tĩnh $0.5 \cdot \mathbf{1}$ (đối chiếu kiểm tra chéo với giá trị đã ghi nhận ở Tuần 1).
   2. Vector ngưỡng tối ưu $\boldsymbol{\theta}^*$ do Vinh bàn giao
- Xuất kết quả ra file trung gian `outputs/tables/test_per_label_f1.csv` hoặc điền trực tiếp vào các cột `f1_base_test`, `f1_opt_test` và tính `delta_f1_test = f1_opt_test − f1_base_test` trong file `per_label_table4.csv`.   
- Lưu ý kiểm tra chuỗi cấm (CI Grep): Script đánh giá của Thy phải đặt trong thư mục `scripts/` (ngoài thư mục `src/`) để lệnh kiểm tra của CI:

``` Bash
grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py
```

vẫn đảm bảo chỉ trả về duy nhất một file là `src/eval/test_gate.py`

**Kiểm tra bắt buộc:**
- Không còn ô trống: `df.notna().all()` đúng.
- Mọi cột F1 khớp `sklearn.metrics.f1_score(average=None, zero_division=0)` ở `1e-9` (phần Test kiểm tra ngay trong `test_gate.py`).
- Trung bình các F1 từng nhãn bằng Macro-F1 mà Vinh hoặc Ngọc báo cáo cho θ* trên Val và Test, sai số ≤ `1e-9`. Trung bình `f1_base_test` phải khớp Test Macro-F1@0.5 trong `baseline_reproduction.md`.
- Bảng giữ đủ cả hai cột `group_freq3` và `is_neutral`.

### 3.3 Kiểm chứng thực nghiệm hành vi dịch chuyển ngưỡng (Head và Tail)

Với mỗi nhóm freq3 (Head, Mid, Tail), Thy tính `Δt_c = t_c* − 0.5`: trung bình, trung vị, độ lệch chuẩn, tỷ lệ nhãn có `t_c* > 0.5`, và hệ số Spearman giữa `n_c` và `t_c*`. Hai dự đoán cần đối chiếu:

- **Dự đoán thứ nhất (từ kế hoạch ban đầu):** nhãn Head tăng ngưỡng để giảm dương tính giả; nhãn Tail hạ ngưỡng (`t_c* < 0.5`) để tăng Recall.
- **Dự đoán thứ hai (từ lý thuyết `pos_weight`):** `pos_weight_c = (N_train − n_c) / n_c` lớn nhất ở nhãn hiếm, nên xác suất của Tail bị đẩy lên mạnh nhất. Điều này nhất quán với Precision (59.08%) thấp hơn Recall (64.40%) ở ngưỡng 0.5. Theo dự đoán này, ngưỡng của Tail có thể **tăng** để cân bằng F1.

Cách báo cáo: nêu rõ cả hai chiều dự đoán, trình bày số liệu đúng như thu được từ θ*, rồi kết luận giả thuyết H1 (độ lệch ngưỡng do hàm mất mát có trọng số) được ủng hộ, ủng hộ một phần hay không được ủng hộ. Không ép số liệu cho khớp một chiều nào.

Lưu ý khi diễn giải:
- Macro-F1 là hàm hằng từng khúc, nhiều ngưỡng khác nhau cho cùng F1. `t_c*` có thể nằm ở mép plateau; viết rõ điều này.
- Các nhóm nhỏ (9, 10, 9 nhãn) nên dùng thống kê mô tả và kiểm định phi tham số, nêu rõ cỡ mẫu, tránh kết luận mạnh.
- Phần phạt chính quy `λ*` kéo θ về `t_g`; tính điều này khi diễn giải độ lệch so với 0.5.
- Với 5 nhãn có F1 thấp nhất: đánh giá riêng tác động của `exclusive_neutral` (bật và tắt) và 4 chế độ fallback bằng cách tính lại F1 từng nhãn với θ* cố định, trên **Val**. Test chỉ dùng cho các cột Bảng 4 ở cấu hình cuối cùng, không dùng để so sánh hay chọn lại cấu hình.

Kết quả viết vào `reports/per_label_findings.md`. Mọi con số dẫn từ `per_label_table4.csv`.

### 3.4 Nội dung nhánh Core cho báo cáo toàn văn
Thy soạn trong `reports/fulltext_core_sections.md` để Ngọc ghép vào báo cáo chính. Ba mục:

1. **Phương pháp luận toán học:**
   - Ánh xạ `expand_theta` cho 4 chế độ.
   - Quy tắc ra quyết định `Ŷ = (P ≥ t)`.
   - Quy tắc Neutral độc quyền: `m_i = max P[i, c]` trên `c ∈ S_i \ {neutral}`; giữ Neutral khi `P[i, neutral] > m_i` (so sánh chặt), loại Neutral khi bằng hoặc nhỏ hơn.
   - 4 chế độ fallback: `argmax` (mặc định), `neutral`, `head_argmax` (argmax trong `head_idx` của `mapping_freq3.json`), `none`.
   - Hàm mục tiêu `J_λ` và ngưỡng toàn cục `t_g`.
2. **Đặc trưng địa hình hàm số:**
   - Phân tích kết quả 1D và 2D.
   - Lý giải vì sao gradient bằng 0 gần như mọi nơi, dựa trên plateau ratio định lượng ở D = 1, 3, 28.
   - Bàn luận số cụm cực trị địa phương từ 200 lượt (D = 3 với 1000 NFE, D = 28 với 2000 NFE, nêu rõ ngân sách riêng của thí nghiệm này).
3. **Phân tích Bảng 4:** biện luận mức cải thiện F1 theo nhóm nhãn và đối chiếu H1 theo mục 3.3.

Yêu cầu: tiếng Việt học thuật, dẫn nguồn (Lipton et al. 2014; Pillai et al. 2013; Demszky et al. 2020), số liệu dẫn từ CSV.

### 3.5 Biểu đồ độ phân giải cao và slide
- Xuất lại 5 biểu đồ địa hình của Tuần 2 và biểu đồ dịch chuyển ngưỡng `t_c*` theo nhóm ở 300 dpi (PNG và SVG hoặc PDF) vào `outputs/figures/hires/`. Dữ liệu địa hình chỉ lấy từ Val.
- Chuẩn bị nội dung slide nhánh Core: kiến trúc xử lý dữ liệu, bề mặt hàm số, thuật toán cục bộ, Bảng 4. Luyện thuyết trình, thống nhất thời lượng với Ngọc.

### 3.6 Làm sạch và đóng gói `03_SUBMISSIONS/Thy/`

```
03_SUBMISSIONS/Thy/
├── README.md          (môi trường, lệnh tái lập kết quả)
├── reports/           baseline_reproduction.md, landscape_analysis.md, per_label_findings.md
├── tables/            landscape_summary.csv, per_label_table4.csv
├── figures/           landscape_1d_global.png, landscape_1d_per_label.png, landscape_2d_head_tail.png,
│                      landscape_local_optima_hist.png, plateau_ratio_bar.png
└── test_logs/         timing_objective.txt, golden_test_output.txt, differential_test_1000.txt
```

- Mã nguồn có docstring đầy đủ.
- Chạy lại toàn bộ test và lệnh grep của CI:

```bash
pytest tests/test_decision.py tests/test_objective.py tests/test_local_optimizers.py tests/test_data_io.py
grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py
```

Lệnh grep chỉ được trả về `src/eval/test_gate.py`.

## 4. Output tuần

- `outputs/tables/per_label_table4.csv` (đầy đủ 100% số liệu 28 nhãn, không còn ô trống), `outputs/tables/test_per_label_f1.csv`.
- `scripts/fill_table4.py` và script đánh giá `scripts/eval_test_per_label.py` (không sửa đổi `src/eval/test_gate.py`).
- `reports/per_label_findings.md` và `reports/fulltext_core_sections.md`.
- `outputs/figures/hires/` (biểu đồ độ phân giải cao).
- Slide nhánh Core cho bộ slide bảo vệ.
- Thư mục `03_SUBMISSIONS/Thy/` hoàn chỉnh.

## 5. Definition of Done (DoD Tuần 4)

- [ ] θ* đã kiểm tra shape, cận, nguồn gốc, và xác nhận không dùng Test.
- [ ] Bảng 4 đủ 28 nhãn, không ô trống, có cả `group_freq3` và `is_neutral`; mọi cột F1 khớp sklearn ở `1e-9`; trung bình khớp Macro-F1 của θ* và của baseline.
- [ ] Dữ liệu Test chỉ được nạp trong `src/eval/test_gate.py`; lệnh grep của CI chỉ trả về file này.
- [ ] Phần biện luận Head và Tail có số liệu minh chứng, nêu cả hai chiều dự đoán và kết luận về H1 theo đúng dữ liệu.
- [ ] Đã đánh giá `exclusive_neutral` và fallback trên 5 nhãn có F1 thấp nhất (trên Val).
- [ ] Ba mục báo cáo toàn văn nhánh Core đã giao cho Ngọc; mọi số liệu dẫn từ CSV.
- [ ] Biểu đồ độ phân giải cao đã xuất; slide nhánh Core sẵn sàng.
- [ ] Mã nguồn sạch, có docstring; `03_SUBMISSIONS/Thy/` hoàn chỉnh; pytest xanh.