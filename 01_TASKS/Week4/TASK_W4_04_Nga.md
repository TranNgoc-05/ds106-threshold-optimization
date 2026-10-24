# TASK_04_Nga_Week4 — Bảng 1, hệ thống biểu đồ, nội dung báo cáo toàn văn & slide bảo vệ

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh Global: hoàn thiện Bảng 1, trực quan hóa, viết nội dung Global/NFLT, chuẩn bị bảo vệ |
| **Owner** | Nga |
| **Reviewers** | Ngọc (Leader, A: tích hợp báo cáo, nghiệm thu), Vinh (Peer Reviewer: thống kê và tái lập) |
| **Deadline** | Cuối Tuần 4: Bảng 1, biểu đồ, bản thảo nội dung và slide. Tổng duyệt bảo vệ ở mốc M6 (ngày do Ngọc ấn định) |
| **Phối hợp** | Vinh (C: cung cấp `stats_mwu_holm.csv`), Ngọc (C: khung báo cáo toàn văn, slide) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Hoàn thiện 100% Bảng 1 (`table1_val_performance.csv`) trên 9 ô `(D, budget)` với Mean, Std, runtime, Success Rate và nhãn ý nghĩa thống kê dựa trên `p_holm` của Vinh.
2. Xuất toàn bộ hệ thống biểu đồ: hội tụ, tốc độ đạt nghiệm, chi phí tính toán.
3. Viết nội dung nhánh Global trong báo cáo toàn văn: Động lực học hội tụ & Áp lực chọn lọc, Phân tích hiệu năng Bảng 1 & NFLT, và phần so sánh nhóm cục bộ/toàn cục.
4. Đóng gói slide, mã nguồn và tài liệu tái lập vào `03_SUBMISSIONS/Nga/`; tham gia tổng duyệt bảo vệ.

---

## 2. Input đầu vào

- `outputs/benchmark_results.csv` đã niêm phong (băm SHA-256 theo `LOCK_FREEZE.json`), `outputs/trajectories/*.npz`, `outputs/benchmark_results_selective.csv`, `outputs/tables/selective_pressure_study.csv`, `outputs/profiling_runtime.csv`.
- Vinh: `outputs/tables/stats_mwu_holm.csv` (cột `family, dimension_D, nfe_budget, algo_a, algo_b, median_a, median_b, U, p_raw, p_holm, A12, rank_biserial, significant, degenerate`) và `nflt_summary.csv`.
- Ngọc: khung báo cáo toàn văn, mẫu slide.
- Tài liệu đọc trước: Hansen et al. (2021), *COCO*; DS106 Chapter 4 và Chapter 6.

**Quy tắc kỹ thuật áp dụng cả tuần**

- **Chỉ đọc** hai file đã băm. Không ghi, không sắp xếp lại tại chỗ, không mở/lưu bằng bảng tính. Mọi bảng và biểu đồ ra file mới.
- **Ngưỡng ý nghĩa α = 0.05** theo `PREREG.md`, dùng cùng giá trị với Vinh.
- **Trung thực khoa học:** kết luận chỉ nêu những gì số liệu cho thấy, dẫn nguồn file `.csv`; nếu kết quả trái dự đoán thì nêu đúng như vậy.
- Cách ly Test: nhánh Global không dùng dữ liệu Test; các bảng của Nga là hiệu năng Validation.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Bảng 1 — `src/analysis/table1.py` → `outputs/tables/table1_val_performance.csv`

- Dữ liệu: `benchmark_results.csv` (1.801 dòng), loại `constant_0.5` khỏi phép thống kê (ghi riêng như dòng tham chiếu cuối bảng nếu cần). Điền **đủ 9 ô** `(D, budget) ∈ {1, 3, 28} × {200, 1000, 5000}`; khi viết báo cáo tập trung D=3 và D=28, D=1 đưa vào phụ lục.
- Với mỗi (thuật toán, D, budget) tính:

  | Cột | Định nghĩa |
  |---|---|
  | `mean`, `std` | Trung bình và độ lệch chuẩn (`ddof=1`) của `best_val_score` (Macro-F1 có chính quy hóa, `J_λ`) qua 20 seeds |
  | `time_seconds_mean` | Trung bình `time_seconds` |
  | `success_rate` | Tỷ lệ seed đạt `best_val_score ≥ 0.99 × Max_Score`, trong đó `Max_Score` là giá trị `best_val_score` lớn nhất trong toàn ô `(D, budget)` (mọi thuật toán, mọi seed) |
  | `is_leader` | Thuật toán dẫn đầu ô theo **trung vị** `best_val_score`, cùng quy ước với `nflt_summary.csv` của Vinh; đồng hạng thì cùng được gán |
  | `p_holm_vs_leader` | `p_holm` của cặp (thuật toán, dẫn đầu) lấy từ **họ khám phá** trong `stats_mwu_holm.csv`; tra không phân biệt thứ tự `algo_a`/`algo_b` |
  | `not_worse_than_leader` | `True` nếu là dẫn đầu hoặc `p_holm_vs_leader ≥ 0.05` |

- Lưu ý khi đọc bảng: `λ` khác nhau theo D (D=1 ép `λ=0`), nên chỉ so sánh trong cùng ô, không so điểm tuyệt đối giữa các D.
- Kiểm tra tự động: bảng không có NaN/ô trống; tổng 10 thuật toán × 9 ô = 90 dòng; `Max_Score` khớp giá trị lớn nhất từ CSV gốc; mọi khóa tra `p_holm` đều tìm thấy.
- Gửi Vinh đối soát chéo số trung vị với `nflt_summary.csv` (cùng định nghĩa dẫn đầu).

### 3.2. Hệ thống biểu đồ — `src/analysis/figures.py`

Chuẩn đồ họa chung: độ phân giải ≥ 300 dpi và bản vector đi kèm (PDF), bảng màu thân thiện với người mù màu, **một bảng màu cố định** gán thuật toán ↔ màu/kiểu đường dùng nhất quán trong mọi hình, cỡ chữ ≥ 8 pt, chú thích rõ đơn vị, không có tiêu đề thừa.

1. **`outputs/figures/convergence_curves_D{1,3,28}.png`:** mỗi hình ứng với một D, ba panel (một panel mỗi budget 200/1000/5000), 10 đường (mọi thuật toán): trung vị của `best_so_far` qua 20 seeds và dải phân vị 25–75, trục NFE **thang logarit**. Dữ liệu từ `.npz`: với job dừng sớm (`actual_nfe < budget`) kéo dài đường bằng giá trị cuối. Mỗi budget là một lượt chạy riêng, không ghép chuỗi giữa các budget.
2. **`outputs/figures/nfe_to_99pct_boxplot.png`:** hộp so sánh số NFE cần để đạt 99% giá trị cuối cùng giữa các thuật toán, nhóm theo họ (cục bộ, gradient, toàn cục/tiến hóa, Bayes, ngẫu nhiên). `nfe_to_99pct` **không nằm** trong 8 cột CSV mà tính hậu kỳ từ `.npz`: số NFE nhỏ nhất mà tại đó `best_so_far ≥ 0.99 × best_score` cuối cùng (với `best_score` là giá trị cuối của `best_so_far`). Xuất kèm `outputs/tables/nfe_to_99pct_values.csv` (khóa job và giá trị) để Vinh dùng cho H4.
3. **`outputs/figures/runtime_vs_nfe.png`:** hai panel: (a) `time_seconds` trung bình theo budget (trục log-log) cho từng thuật toán và D; (b) `ms/NFE`. Làm nổi bật chi phí phụ của sampler TPE (so với đường tuyến tính theo NFE của các thuật toán còn lại) và kết hợp thời gian trung bình mỗi trial đo ở Tuần 2.
4. `outputs/figures/premature_convergence_de_pso.png` đã xuất ở Tuần 3: rà lại chất lượng đồ họa và bảng màu cho thống nhất.

### 3.3. Nội dung nhánh Global trong báo cáo toàn văn

Giao cho Ngọc bản thảo học thuật (Markdown), mỗi con số dẫn nguồn file `.csv` và đường dẫn bảng.

**(a) Động lực học hội tụ & Áp lực chọn lọc (Chương 6).**
- Hiện tượng mất độ đa dạng quần thể và `diversity_collapse_nfe` của 8 biến thể (từ `selective_pressure_study.csv`).
- So sánh đột biến `best1` / topology `gbest` / `w` nhỏ với `rand1` / `ring` / `w` lớn: điểm số, thời điểm collapse, khoảng cách tới nghiệm tốt nhất. Nêu cả những kết quả trái dự đoán.
- Giải thích vai trò của quy tắc chọn lọc `≥` trên plateau của hàm bậc thang đối với `diversity`.

**(b) Phân tích hiệu năng Bảng 1 & Định lý No Free Lunch (Chương 4).**
- Dùng Bảng 1, `nflt_summary.csv` và `stats_mwu_holm.csv` để mô tả hoán đổi thứ hạng giữa các thuật toán theo D và budget.
- Kết luận về việc có hay không một thuật toán thống trị mọi ô **chỉ theo số liệu**. Nếu một thuật toán dẫn đầu mọi ô, báo cáo ghi đúng như vậy và nêu giới hạn của kết luận.
- Vai trò Random Search làm mốc đối chứng: ô nào thuật toán không vượt được `random_search` (`p_holm`, `A12` họ chính của Vinh).
- Chi phí thời gian thực so với NFE (chi phí sampler TPE, thời gian trung bình mỗi trial).

**(c) So sánh nhóm cục bộ và toàn cục (Chương 5, mục ngắn).** Mô tả vị trí các thuật toán toàn cục (RS, SA, DE, PSO, TPE) so với nhóm cục bộ và gradient trong Bảng 1, chỉ dựa trên số liệu có sẵn, không suy diễn cơ chế khi chưa có bằng chứng; phần lý giải tính gần tách biến thuộc H3 trong `eval_findings.md` của Vinh, Nga chỉ tham chiếu.

### 3.4. Slide thuyết trình và tổng duyệt (M6)

- Chuẩn bị slide (khoảng 6–8 trang, theo mẫu của Ngọc): (1) bài toán và vai trò Random Search trong NFLT; (2) 5 thuật toán toàn cục và quyết định thiết kế (SA plateau, DE `≥` và phản xạ, PSO Clerc–Kennedy); (3) Batch Runner (Single Writer, `--resume`, tái lập bit, `fsync`); (4) công bằng siêu tham số LD-7 và K-5; (5) Bảng 1 và đường cong hội tụ; (6) hội tụ sớm và áp lực chọn lọc; (7) NFLT và chi phí thời gian thực; (8) hạn chế.
- Tham gia tổng duyệt; chuẩn bị trả lời phản biện về thuật toán tiến hóa và chi phí tính toán: tại sao `≥` thay vì `>`, tại sao tự cài DE thay vì dùng SciPy, vai trò của `NP` theo budget, tại sao TPE chi phí thật cao hơn NFE, tại sao `multivariate=False` ở D=28 (K-5), tính tái lập từng bit được bảo đảm thế nào.

### 3.5. Đóng gói `03_SUBMISSIONS/Nga/`

```text
03_SUBMISSIONS/Nga/
├── README.md                              # môi trường, lệnh tái lập toàn bộ benchmark
├── reports/{global_algorithms.md, selective_pressure_study.md}
├── tables/{tuning_log.csv, selective_pressure_study.csv, table1_val_performance.csv,
│           benchmark_results.csv, sha256.txt}
├── figures/{convergence_curves_D1.png, convergence_curves_D3.png, convergence_curves_D28.png,
│            nfe_to_99pct_boxplot.png, premature_convergence_de_pso.png, runtime_vs_nfe.png}
└── test_logs/{run_log.txt, reproducibility_test_log.txt, validate_results_output.txt}
```

- `tables/benchmark_results.csv` là **bản sao nguyên trạng** của file đã băm, kèm `sha256.txt` (mã băm tính lại bằng `sha256sum`, so khớp `LOCK_FREEZE.json`).
- README ghi: phiên bản môi trường, lệnh tái lập lưới chính, lưới tuning, nghiên cứu áp lực chọn lọc, thứ tự CPU, cách chạy `validate_results`.
- Tự rà soát và tích đủ 10 tiêu chuẩn nghiệm thu PR (Khóa Interface, `BaseOptimizer`, bộ đếm NFE công bằng, grep CI, tái lập bit, kiểm thử đơn vị, đủ 1.801 dòng, đủ file bên cạnh, công bằng siêu tham số, nghiên cứu hội tụ sớm).

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Bảng | `outputs/tables/table1_val_performance.csv`, `outputs/tables/nfe_to_99pct_values.csv` |
| Biểu đồ | `outputs/figures/convergence_curves_D1.png`, `convergence_curves_D3.png`, `convergence_curves_D28.png`, `nfe_to_99pct_boxplot.png`, `runtime_vs_nfe.png` (và `premature_convergence_de_pso.png` đã rà soát) |
| Mã nguồn | `src/analysis/table1.py`, `src/analysis/figures.py` |
| Văn bản | Bản thảo Chương 4 (NFLT, Bảng 1), Chương 5 (so sánh cục bộ/toàn cục), Chương 6 (áp lực chọn lọc) cho Ngọc |
| Slide | Bộ slide nhánh Global theo mẫu của Ngọc |
| Đóng gói | `03_SUBMISSIONS/Nga/` |

---

## 5. Definition of Done (DoD tuần)

- [ ] Bảng 1 điền đủ 100% (90 dòng, không NaN), có `success_rate`, `time_seconds_mean`, `not_worse_than_leader` từ `p_holm` (α = 0.05) của Vinh; trung vị khớp `nflt_summary.csv`.
- [ ] Đủ biểu đồ `convergence_curves_D{1,3,28}.png`, `nfe_to_99pct_boxplot.png`, `runtime_vs_nfe.png`, `premature_convergence_de_pso.png` theo chuẩn đồ họa xuất bản (≥ 300 dpi, bảng màu nhất quán).
- [ ] `nfe_to_99pct` được tính hậu kỳ từ `.npz` theo đúng định nghĩa đã chốt.
- [ ] Bản thảo Chương 4, 5, 6 đã bàn giao Ngọc; mọi con số dẫn nguồn `.csv`; kết quả trái dự đoán (nếu có) được nêu rõ.
- [ ] Hai file đã băm không bị sửa (`sha256sum` khớp `LOCK_FREEZE.json`); mọi bảng và biểu đồ ra file mới.
- [ ] Slide hoàn chỉnh, Nga tham gia tổng duyệt (M6) và chuẩn bị đủ câu trả lời phản biện.
- [ ] Đóng gói `03_SUBMISSIONS/Nga/` hoàn chỉnh, `pytest tests/test_global_optimizers.py tests/test_runner.py` đạt 100% PASS.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py`.
- [ ] Ngọc nghiệm thu và tích hợp vào báo cáo toàn văn.