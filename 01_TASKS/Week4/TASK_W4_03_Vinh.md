# TASK_03_Vinh_Week4 — Đánh giá Test (Bảng 2), Ablation (Bảng 3), Kiểm định MWU-Holm & Báo cáo cuối kỳ

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh EVAL: đánh giá tổng quát hóa, ablation, kiểm định thống kê và kết luận giả thuyết |
| **Owner** | Vinh |
| **Reviewers** | Trần Khánh Ngọc (Leader, A: nghiệm thu M5, tích hợp báo cáo chính), Thy (Peer: Bảng 1 và 4, đối chiếu tái lập), Nga (C: hỗ trợ runner) |
| **Deadline** | Cuối Tuần 4, tương ứng mốc M5 (Test, ablation, thống kê) |
| **Phối hợp** | Ngọc (cấp `LOCK_FREEZE.json`), Nga (runner bổ sung và ablation), Thy (`f1_base_val`, Bảng 4) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Tiếp nhận `LOCK_FREEZE.json` từ Ngọc (sau Cổng G4), trích xuất và **đóng băng** θ\* của Bảng 2 và Bảng 3 **trước** khi gọi `load_test()` lần đầu.
2. Mở cổng Test, hoàn thiện **Bảng 2** (tổng quát hóa) và bảng phân tán theo seed.
3. Thực thi Ablation, hoàn thiện **Bảng 3**.
4. Chạy kiểm định Mann-Whitney U + Holm + A12 trên 9 ô `(D, budget)` và tổng hợp kiểm chứng NFLT.
5. Soạn `reports/eval_findings.md` (≤ 3 trang) trả lời H1–H4 và NFLT theo số liệu thực, bàn giao cho Ngọc.

---

## 2. Input đầu vào

- Ngọc: `artifacts/LOCK_FREEZE.json` (chỉ băm đúng `outputs/benchmark_results.csv` 1.801 dòng và `outputs/benchmark_theta_log.jsonl`).
- Nga: kho benchmark chính `outputs/benchmark_results.csv`, `outputs/benchmark_theta_log.jsonl` (chỉ đọc), quỹ đạo `.npz`, runner.
- Thy: `f1_base_val` (F1 Val từng nhãn ở ngưỡng 0.5), `make_objective`, `compute_t_g`; Bảng 1 và Bảng 4.
- Vinh (các tuần trước): `artifacts/lambda_star.json`, `outputs/sepcheck_results.csv`, `optimality_gap_D28.csv`, `test_gate.py`, `separable_oracle.py`.
- `00_START_HERE/PREREG.md` (giả thuyết và tiêu chí đã đăng ký).
- Đọc trước: DS106 Chapter 4 *No Free Lunch Theorem* (pp. 1–8); Wolpert & Macready (1997); Holm (1979); Vargha & Delaney (2000).

**Quy tắc kỹ thuật áp dụng cả tuần**

- **Không sửa 2 file đã băm.** Lượt chạy bổ sung (D=4, D=28 λ=0) ghi vào `outputs/supplementary_results.csv`. Ablation ghi vào `outputs/ablation_results.csv`.
- **Thứ tự bắt buộc:** trích xuất và đóng băng toàn bộ θ\* (Bảng 2 và 3) vào `artifacts/theta_frozen/` **trước** lần `load_test()` đầu tiên. Dry-run với `--use_mock` được phép bất kỳ lúc nào.
- **Cách ly Test:** chỉ `test_gate.py` được nhắc tới `P_test`, `Y_test`, `_test.npy`. `generalization.py`, `ablation.py` chỉ gọi `load_test()` và đặt biến với tên khác (ví dụ `P_t`, `Y_t`).
- **K-3:** phạt L2 là regularization; coupling đến từ Neutral độc quyền và Fallback.
- **Trung thực khoa học:** không chỉnh số liệu, không khẳng định trước kết quả, mỗi kết luận dẫn nguồn từ file `.csv`.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Tiếp nhận niêm phong và chuẩn bị (chưa mở cổng)

1. Nhận `LOCK_FREEZE.json` từ Ngọc. **Chưa** gọi `load_test()`.
2. Kiểm tra thủ công băm bằng CLI, không qua `load_test()`:

   ```bash
   sha256sum outputs/benchmark_results.csv outputs/benchmark_theta_log.jsonl
   ```

   So với chuỗi trong `LOCK_FREEZE.json`. Nếu lệch, báo Ngọc tính lại trên Linux/CI.
3. Dry-run toàn bộ pipeline với `--use_mock` để bắt lỗi sớm.

### 3.2. Lượt chạy bổ sung (chỉ Validation, ghi vào file riêng)

- Hai cấu hình không có trong benchmark chính: **D=4 (`semantic4`)** và **D=28, λ=0**. Lệnh minh họa (đối chiếu cờ theo README runner):

  ```bash
  python -m src.runner.run_benchmark --mapping semantic4 --dims 4 --lam-file artifacts/lambda_star.json --budgets 5000 --seeds 0-19 --out outputs/supplementary_results.csv
  
  python -m src.runner.run_benchmark --mapping per_label  --dims 28 --lam 0 --budgets 5000 --seeds 0-19 --out outputs/supplementary_results.csv
  ```
- NFE = 5000, 20 seeds (`0–19`). Với D=4, dùng `λ = λ*_3` cho nhất quán với Ablation (không có CV riêng cho D=4). Ghi giả định này vào báo cáo.
- Phạm vi thuật toán (đề xuất: cả 10 thuật toán để định nghĩa "nghiệm tốt nhất toàn cấu hình" ở 3.3 đầy đủ, tối thiểu `coordinate_ascent` + `differential_evolution` nếu thiếu tài nguyên). Ngọc xác nhận.

### 3.3. Trích xuất θ\* trên Validation và đóng băng

Với mỗi cấu hình, `θ*` là nghiệm có `best_val_score` cao nhất trong **toàn bộ lần chạy ở NFE = 5000** của cấu hình đó (chỉ dùng Val; hòa điểm chọn nghiệm gần `t_g` nhất). 6 cấu hình của Bảng 2:

| # | Cấu hình | Nguồn |
|---|---|---|
| 1 | Baseline tĩnh: ngưỡng cố định 0.5 | Hằng số |
| 2 | D=1 (global), `t_g` | `benchmark_*` |
| 3 | D=3 `freq3`, `λ*_3` | `benchmark_*` |
| 4 | D=4 `semantic4` | `supplementary_results.csv` |
| 5 | D=28, λ=0 | `supplementary_results.csv` |
| 6 | D=28, `λ*_28` | `benchmark_*` |

Đóng băng vào `artifacts/theta_frozen/` (JSON, kèm khóa nguồn: file nguồn, thuật toán, seed, NFE, `best_val_score`, cấu hình, git hash):

- `table2_<config>.json` cho 6 cấu hình trên.
- `table3_<config_id>_<algo>.json` cho từng cặp cấu hình ablation × thuật toán, chứa `theta_star` và `theta_per_seed` (20 × D) để tính trung bình ± độ lệch chuẩn ở 3.5.
- Một `manifest.json` liệt kê toàn bộ file kèm SHA-256 của chính chúng.

Bảng phân tán theo seed (`test_generalization_by_algo.csv`) dùng 20 nghiệm mỗi (thuật toán, D) lấy trực tiếp từ `benchmark_theta_log.jsonl` (đã băm) và `supplementary_results.csv`, nên không cần đóng băng riêng.

**Bàn giao Bảng 4:** gửi Thy vector `θ*` của cấu hình D=28, `λ*` (`table2_D28_lamstar.json`). **Thy** tính và điền `per_label_table4.csv` (`t_c*`, F1 Val/Test trước 0.5 và sau θ\*, `ΔF1`). Khi cần số Test, Thy gọi `load_test()` sau khi cổng mở và tuân thủ cách ly (không dùng các chuỗi bị cấm trong `src/`).

### 3.4. Bảng 2: đánh giá tổng quát hóa — `src/eval/generalization.py`

Sau khi đã đóng băng xong, mở cổng bằng `test_gate.load_test()` (lần đầu tiên; thất bại → `PermissionError`, dừng và báo Ngọc).

Với từng trong 6 cấu hình, tính trên θ\* đã đóng băng, theo quy tắc quyết định `full` (Neutral độc quyền + Fallback argmax):

| Cột | Định nghĩa |
|---|---|
| Macro-F1 (Val), Macro-F1 (Test) | Macro-F1 **không phạt** của θ\* |
| Generalization Gap | `Macro-F1(Val) − Macro-F1(Test)` |
| Weighted-F1 (Test) | `sklearn.metrics.f1_score(..., average="weighted", zero_division=0)` |
| F1 nhóm 5 nhãn yếu nhất (Test) | Trung bình F1 Test của 5 nhãn có F1 Val thấp nhất ở ngưỡng 0.5 |

**5 nhãn yếu nhất:** tính **động** từ `f1_base_val` của Thy (5 nhãn có F1 Val thấp nhất ở ngưỡng 0.5), cố định cho mọi cấu hình. Đối chiếu với danh sách Word (*disapproval, neutral, realization, desire, confusion*). Nếu lệch, dùng kết quả động và ghi chú sự khác biệt trong báo cáo.

**Bảng phân tán theo seed:** với mỗi thuật toán ở NFE=5000 và mỗi D, tính Macro-F1 Test của 20 nghiệm (mean ± std, và gap). Xuất `outputs/tables/test_generalization_by_algo.csv`.

**Xuất:**

- `outputs/tables/table2_generalization.csv` (không NaN, không ô trống).
- `outputs/tables/test_generalization_by_algo.csv`.
- `outputs/figures/generalization_gap_bar.png` (cột Generalization Gap theo cấu hình).

### 3.5. Bảng 3: Ablation — `src/eval/ablation.py`

Tối ưu bằng 2 thuật toán `coordinate_ascent` (đại diện cục bộ) và `differential_evolution` (đại diện toàn cục), NFE=5000, seeds `0–19`, `t_g` cố định (LD-9). **9 cấu hình × 2 thuật toán × 20 seeds = 360 lượt**, tách D=3 và D=4:

| `config_id` | Mô tả | D | Tham số |
|---|---|---|---|
| `full` | Neutral độc quyền + Fallback argmax + `λ*_28` | 28 | `fallback=argmax`, `excl=True`, `λ=λ*_28` |
| `no_fallback` | Chấp nhận dự đoán rỗng | 28 | `fallback=none` |
| `fb_neutral` | Fallback về Neutral | 28 | `fallback=neutral` |
| `fb_head` | Fallback về nhãn Head có xác suất cao nhất | 28 | `fallback=head_argmax` |
| `no_excl_neutral` | Bỏ Neutral độc quyền | 28 | `excl=False` |
| `lam0` | Bỏ phạt L2 | 28 | `λ=0` |
| `freq3` | Phân nhóm tần suất | 3 | `mapping=freq3`, `λ=λ*_3` |
| `semantic4` | Phân nhóm ngữ nghĩa | 4 | `mapping=semantic4`, `λ=λ*_3` |
| `sep_check` | Tách biến thuần | 28 | `fallback=none`, `excl=False`, `λ=λ*_28` |

**Tái sử dụng kết quả khi trùng cấu hình** (đồng nhất cấu hình, NFE, seeds, `t_g`):

- `full`, `freq3` ← `benchmark_results.csv` (chỉ đọc).
- `lam0`, `semantic4` ← `supplementary_results.csv`.
- `sep_check` ← `sepcheck_results.csv` (Tuần 3, hai thuật toán CA và DE ở NFE=5000).
- Chạy mới: `no_fallback`, `fb_neutral`, `fb_head`, `no_excl_neutral` (4 × 2 × 20 = 160 lượt).
- Ghi nguồn gốc từng khối kết quả vào `outputs/ablation_provenance.json` (giữ nguyên schema 8 cột chuẩn + `config_id` của `ablation_results.csv`).
- Trước khi tái sử dụng, kiểm tra tự động `λ` dùng thực tế bằng `λ*` trong `lambda_star.json` và `t_g` bằng `t_g.json`.

**Đánh giá:** Macro-F1 không phạt trên Val và Test theo **quy tắc quyết định của chính cấu hình đó** (ví dụ `no_fallback` đánh giá không Fallback). Báo cáo `mean ± std` trên 20 seeds cho Val và Test, `Gap`, và chênh lệch so với `full` (`ΔVal`, `ΔTest`, `ΔGap`). Vì quy tắc quyết định khác nhau giữa các biến thể nên chỉ so sánh chênh lệch một cách thận trọng, không suy diễn nhân quả ngoài số liệu.

Chỉ kết luận theo K-3: so sánh `lam0` với `full` để định lượng vai trò chống quá khớp của phạt L2 (nghiệm co về `t_g`, gap Val−Test thay đổi ra sao). Mức đóng góp coupling được đọc qua `no_fallback`, `no_excl_neutral`, `sep_check`.

**Xuất:** `outputs/ablation_results.csv`, `outputs/tables/table3_ablation.csv`, `outputs/figures/ablation_table3.png`.

### 3.6. Kiểm định thống kê — `src/eval/stats_mwu_holm.py`

**Dữ liệu:** `benchmark_results.csv` (1.801 dòng), loại dòng `constant_0.5` (n=1). Còn 1.800 dòng = 9 ô × 10 thuật toán × 20 seeds.

**Phân vùng:** 9 ô `(D, budget) ∈ {1, 3, 28} × {200, 1000, 5000}`.

**Hai họ kiểm định, Holm hiệu chỉnh riêng trong từng ô:**

1. **Họ chính (NFLT):** so từng thuật toán trong 9 thuật toán còn lại với `random_search`, `m = 9` mỗi ô.
2. **Họ khám phá:** tất cả cặp thuật toán, `m = C(10,2) = 45` mỗi ô.
- Tổng số dòng đầu ra: `9 ô × (9 + 45) = 486`.

**Kiểm định:**

```python
from scipy.stats import mannwhitneyu
res = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
```

- Hàm bậc thang tạo nhiều giá trị trùng nên không dùng `exact`.
- Nếu hai mẫu hoàn toàn giống nhau (U suy biến) → đặt `p_raw = 1.0`, `degenerate = True`, không để chương trình lỗi hoặc ra NaN.
- Hiệu chỉnh Holm thủ công (Holm–Bonferroni từng bước) trong từng (họ, ô).
- **Cỡ hiệu ứng:**

  ```text
  A12 = U_a / (n_a · n_b)           # = P(a > b) + 0.5·P(a = b), U_a là thống kê U của mẫu a (hướng "lớn hơn tốt hơn")
  rank_biserial = 2·A12 − 1
  ```
- `significant = (p_holm < α)` với `α` lấy theo `PREREG.md` (mặc định 0.05 nếu không quy định khác).
- Cột `stats_mwu_holm.csv`: `family, dimension_D, nfe_budget, algo_a, algo_b, median_a, median_b, U, p_raw, p_holm, A12, rank_biserial, significant, degenerate`.

**Đối soát độ chính xác** (`tests/test_stats.py`, dữ liệu giả):

- `p_holm` khớp `statsmodels.stats.multitest.multipletests(..., method="holm")` với sai số `≤ 1e-12`. Chỉ dùng statsmodels để đối soát **p-Holm**.
- `A12` đối soát qua công thức quan hệ với `U`: `A12 = U_a/(n_a n_b)` và `rank_biserial = 2·A12 − 1`.
- Ca suy biến (hai mẫu đồng nhất) trả về `p = 1.0`, `degenerate = True`.
- Lưu log vào `03_SUBMISSIONS/Vinh/test_logs/stats_verification_log.txt`.

**Tổng hợp NFLT — `outputs/tables/nflt_summary.csv`** (mỗi dòng một thuật toán):

- `n_cells_best`: số ô thuật toán đứng nhất theo **trung vị** `best_val_score` (đồng hạng thì tính cho mọi thuật toán cùng hạng, ghi chú quy ước này).
- `n_cells_not_worse_than_best`: số ô thuật toán không kém thuật toán dẫn đầu một cách có ý nghĩa (dựa trên họ khám phá, `p_holm ≥ α` so với thuật toán dẫn đầu, hoặc chính là thuật toán dẫn đầu).
- `n_cells_worse_than_rs`: số ô tệ hơn `random_search` có ý nghĩa (họ chính, `p_holm < α` và `A12 < 0.5`); thêm cột `n_cells_better_than_rs` để đối xứng.
- Tổng số ô mỗi thuật toán: 9.

**Biểu đồ:** `outputs/figures/holm_pvalue_heatmap_D3.png` và `holm_pvalue_heatmap_D28.png`: mỗi hình 3 panel (một budget mỗi panel), ma trận 10 × 10 màu theo `−log10(p_holm)`, đánh dấu ô có ý nghĩa.

**Bàn giao Nga:** `p_holm` cho Bảng 1 (Nga dựng bảng, Vinh cấp thống kê).

### 3.7. Báo cáo — `reports/eval_findings.md` (tối đa 3 trang)

Đối chiếu số liệu thực nghiệm với các tiêu chí định lượng đã đăng ký trong `00_START_HERE/PREREG.md`. Mỗi mục có **một dòng kết luận**:

> **Kết luận: Ủng hộ / Không ủng hộ / Chưa đủ bằng chứng**, kèm số trích dẫn và đường dẫn bảng.

Không khẳng định trước khi có số liệu. Kết quả trái dự đoán phải được nêu rõ. Ngưỡng quyết định đọc từ `PREREG.md`, không tự đặt.

| Mục | Nội dung báo cáo | Nguồn số liệu |
|---|---|---|
| **H1** (độ lệch ngưỡng) | Giá trị cụ thể của `t_g`; so sánh Macro-F1 tại `t_g` với mốc 0.5 trên Validation | `t_g.json`, Bảng 2, `f1_base_val` |
| **H2** (quá khớp ở 28 chiều) | Generalization Gap của D=28 λ=0, D=28 `λ*`, D=3, D=1; việc thêm phạt `λ*` hoặc rút gọn về D=3 có thu hẹp gap Val−Test không; phạt L2 là regularization (K-3) | `table2_generalization.csv`, `test_generalization_by_algo.csv`, `table3_ablation.csv` (`lam0` vs `full`) |
| **H3** (CA và tính gần tách biến) | Kết quả Holm của `coordinate_ascent` so với các thuật toán khác theo từng D và budget; Optimality Gap ở D=28; phân tích thuật toán quần thể chỉ vượt ở budget nhỏ hoặc không gian gộp (nếu số liệu hỗ trợ); đóng góp coupling qua `no_fallback`, `no_excl_neutral`, `sep_check` | `stats_mwu_holm.csv`, `optimality_gap_D28.csv`, `table3_ablation.csv` |
| **H4** (độ lệch của gradient) | Tốc độ hội tụ `nfe_to_99pct` của Adam/L-BFGS-B so với hộp đen; `is_converged` sớm của L-BFGS-B; **định lượng độ lệch nghiệm** theo hai thành phần: (i) tại `sep_check` (không Neutral/Fallback) gap còn lại chủ yếu do `τ`; (ii) tại cấu hình `full` gap bao gồm cả `τ` lẫn Neutral/Fallback không mô hình hóa | `benchmark_results.csv`, `optimality_gap_D28.csv`, `surrogate_notes.md` |
| **NFLT** | Hoán đổi thứ hạng giữa các thuật toán qua D và budget: số ô đứng nhất của mỗi thuật toán, không thuật toán nào dẫn đầu mọi ô (hoặc ngược lại nếu số liệu cho thấy vậy), số ô không kém hơn nhất, số ô tệ hơn Random Search | `nflt_summary.csv`, `stats_mwu_holm.csv` |

- Mọi con số trích từ `.csv` trong `outputs/tables/`, kèm đường dẫn file.
- Bàn giao toàn bộ bảng, biểu đồ và văn bản cho Ngọc để tổng hợp báo cáo toàn văn.

### 3.8. Đóng gói và rà soát PR

- Đóng gói `03_SUBMISSIONS/Vinh/` gồm `README.md` (lệnh tái lập toàn bộ), `reports/`, `tables/`, `figures/`, `test_logs/`.
- Tự rà soát 10 tiêu chuẩn nghiệm thu PR của Word trước khi gửi Ngọc, trong đó lệnh grep là lệnh CI của Ngọc.
- Chạy `pytest tests/test_surrogate_gradcheck.py tests/test_gradient_optimizers.py tests/test_separable_oracle.py tests/test_test_gate.py tests/test_stats.py` đạt 100% PASS.

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Mã nguồn | `src/eval/generalization.py`, `ablation.py`, `stats_mwu_holm.py` |
| Test | `tests/test_stats.py` |
| Dữ liệu chạy | `outputs/supplementary_results.csv`, `outputs/ablation_results.csv`, `outputs/ablation_provenance.json` |
| Artifact | `artifacts/theta_frozen/*.json` (6 file Bảng 2, các file Bảng 3, `manifest.json`) |
| Bảng | `outputs/tables/table2_generalization.csv`, `test_generalization_by_algo.csv`, `table3_ablation.csv`, `stats_mwu_holm.csv`, `nflt_summary.csv` |
| Biểu đồ | `outputs/figures/generalization_gap_bar.png`, `ablation_table3.png`, `holm_pvalue_heatmap_D3.png`, `holm_pvalue_heatmap_D28.png` |
| Log | `03_SUBMISSIONS/Vinh/test_logs/stats_verification_log.txt` |
| Báo cáo | `reports/eval_findings.md` (≤ 3 trang) |
| Bàn giao | Vector θ\* (D=28, `λ*`) cho Thy; `p_holm` cho Bảng 1 của Thy |
| Đóng gói | `03_SUBMISSIONS/Vinh/` |

---

## 5. Definition of Done (DoD tuần)

- [ ] Toàn bộ θ\* của Bảng 2 và Bảng 3 đã đóng băng vào `artifacts/theta_frozen/` (có `manifest.json`) **trước** lần `load_test()` đầu tiên.
- [ ] Cổng Test mở hợp lệ qua `test_gate.load_test()` với SHA-256 khớp 100%. Hai file benchmark đã băm không bị sửa. Lượt chạy bổ sung nằm ở file riêng.
- [ ] Bảng 2 và Bảng 3 điền đủ 100%, không ô trống, không NaN. Bảng phân tán theo seed hoàn chỉnh.
- [ ] 5 nhãn yếu nhất tính động từ `f1_base_val`, đã đối chiếu và ghi chú so với danh sách Word.
- [ ] Ablation đủ 9 cấu hình × 2 thuật toán × 20 seeds, có `ablation_provenance.json` ghi rõ phần tái sử dụng.
- [ ] MWU + Holm chạy trên đủ 9 ô, đủ 486 dòng (9 × (9 + 45)). `p_holm` khớp `statsmodels` ở `≤ 1e-12`. `A12` khớp công thức quan hệ với `U`. Ca suy biến được xử lý (`p = 1.0`, `degenerate = True`).
- [ ] `nflt_summary.csv` và `holm_pvalue_heatmap_D{3,28}.png` được xuất.
- [ ] `eval_findings.md` ≤ 3 trang, mỗi H1–H4 và NFLT có một dòng "Ủng hộ / Không ủng hộ / Chưa đủ bằng chứng" kèm số trích dẫn và đường dẫn bảng. Kết quả trái dự đoán được nêu rõ, không chỉnh số liệu.
- [ ] Vector θ\* (D=28, `λ*`) đã bàn giao Thy để điền Bảng 4. `p_holm` đã gửi Thy cho Bảng 1.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py`.
- [ ] `pytest` toàn bộ 5 file test đạt 100% PASS. `03_SUBMISSIONS/Vinh/` đóng gói hoàn chỉnh. Ngọc nghiệm thu M5.