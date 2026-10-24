# TASK_02_Thy_Week2 — Ba thuật toán cục bộ, khảo sát địa hình toàn diện và Cổng G2

| Mục | Nội dung |
|---|---|
| Vai trò | Nhánh Core: tối ưu hóa cục bộ và khảo sát bề mặt hàm số |
| Owner | Thy |
| Reviewers | Ngọc (Leader, nghiệm thu và kiểm toán NFE Cổng G2), Vinh (Peer, nhất quán hàm mục tiêu) |
| Phối hợp | Nga (runner `run_benchmark.py`) |
| Mã task gốc | CORE-01 |
| Khung thời gian | Tuần 2 trong kế hoạch 4 tuần, mỗi tuần 7 ngày lịch |
| Deadline tuần | 23:30 ngày cuối Tuần 2 |

## 1. Mục tiêu tuần (Weekly Goal)

1. Hoàn thiện và đóng gói 3 thuật toán cục bộ `coordinate_ascent`, `nelder_mead`, `powell` theo chuẩn `BaseOptimizer`.
2. Cài `evaluate_constant` (baseline tĩnh tại 0.5).
3. Hoàn thành khảo sát địa hình B1 đến B4 trên Val thật.
4. Tinh chỉnh siêu tham số theo LD-7.
5. Vượt Cổng G2 (kiểm toán NFE) của Ngọc.

## 2. Input đầu vào

- `BaseOptimizer` và `NFETracker` từ `src/contract/`.
- `P_val.npy`, `Y_val.npy`, `artifacts/t_g.json`, `mapping_freq3.json`, `mapping_semantic4.json` từ Tuần 1.
- Quy tắc công bằng LD-7: tối đa 5 cấu hình, seeds 100–104.
- Tài liệu bắt buộc đọc:
  - Slide DS106 Chapter 1 *Function Optimization* (pp. 1–10) và Chapter 5 *Local and Global Optimization* (pp. 1–12).
  - Nelder & Mead (1965), Powell (1964), Gao & Han (2012).
  - Mục tiêu: nắm khái niệm bề mặt phản hồi, hiểu vì sao Macro-F1 là hàm bậc thang có gradient bằng 0, và hiểu cơ chế co cửa sổ của Coordinate Ascent.

**Ràng buộc xuyên suốt**
- Chỉ cài `_search`, tuyệt đối không ghi đè `optimize()`. Mọi đánh giá hàm mục tiêu đều qua `tracker`.
- Chỉ dùng Val qua `data_io.load_split("val")`. Không nạp, import hay nhắc đến dữ liệu Test trong `src/optimizers/` và `landscape.py` (kể cả comment, docstring).
- Lệnh grep của CI phải sạch: `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py`.

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 `coordinate_ascent.py`
Tham số: `K = 7`, `w0 = 0.45`, hệ số co `0.5`, `w_min = 0.002`. Điểm xuất phát `x0` do `optimize()` cấp (LD-5).

```
w = w0
lặp:
    cải_thiện = False
    với j = 0 .. D-1 (thứ tự tuần hoàn):
        lo = max(LO, x[j] - w); hi = min(HI, x[j] + w)
        lưới = K điểm đều trong [lo, hi]; điểm gần x[j] nhất được thay bằng chính x[j]
        đánh giá K-1 điểm mới qua tracker (không đánh giá lại điểm hiện tại)
        nếu J_tốt_nhất > J_hiện_tại (tăng chặt): x[j] = điểm tốt nhất; cải_thiện = True
        ngược lại (kể cả hòa): giữ nguyên x[j]
    nếu w <= w_min và không cải_thiện: dừng, trả True
    w = w * 0.5
```

- Một lượt quét tốn `D·(K−1)` NFE (D = 28 thì 168). Ở NFE = 200 chỉ hoàn thành khoảng 1,2 lượt. Đây là hành vi mong đợi, không phải lỗi.
- Chỉ nhận nghiệm khi `J` tăng chặt, để không trôi ngẫu nhiên trên plateau.
- Cần một lần `tracker(x0)` để có `J(x0)`, trừ khi `BaseOptimizer` đã cấp sẵn giá trị. Theo đúng cách `base_optimizer.py` định nghĩa.
- Điểm lưới bị kẹp biên có thể không chứa `x[j]`. Khi đó thay điểm lưới gần `x[j]` nhất bằng `x[j]` để luôn đúng `K−1` NFE mỗi tọa độ. Ghi quyết định này vào docstring.

### 3.2 `nelder_mead.py`
- Dùng `scipy.optimize.minimize(method="Nelder-Mead")` trên `lambda x: -tracker(x)`, `bounds=[(LO, HI)]*D`, `x0` do `optimize()` cấp.
- `initial_simplex`: `x0` cộng `0.1·e_j` (đảo chiều nếu vượt cận).
- `adaptive=True` khi `D ≥ 10`.
- `options={"maxfev": tracker.remaining, "xatol": 1e-3, "fatol": 1e-5}`. Lý do `fatol=1e-5`: một mẫu lật nhãn hiếm đổi Macro-F1 cỡ `2e-4` đến `4e-4`, nên `fatol` phải nhỏ hơn mức này để không dừng giả.
- `BudgetExhausted` do `optimize()` xử lý. Không `try/except` nuốt lỗi khác.
- Ghi nhận có chủ đích: trên hàm bậc thang simplex dễ dẹt (mọi đỉnh cùng giá trị) và dừng sớm. Đây là bằng chứng hội tụ sớm (`is_converged=True`).

### 3.3 `powell.py`
- `scipy.optimize.minimize(method="Powell")`, `bounds=[(LO, HI)]*D`, `options={"maxfev": tracker.remaining, "xtol": 1e-3, "ftol": 1e-6}`.
- `maxfev` của SciPy chỉ gần đúng. Bộ đếm chính xác thuộc `tracker` (ngắt bằng `BudgetExhausted`). Không dựa vào `maxfev`.
- Powell là ca dễ lọt NFE nhất khi review G2, nên cần test riêng chứng minh `actual_nfe ≤ max_nfe`.

### 3.4 `evaluate_constant(P, Y, lam, t_g)`
Trả `J_λ` tại `θ = 0.5·1` bằng đúng **một** lần gọi (1 NFE). Vì θ là hằng nên phần phạt bằng `λ·(0.5 − t_g)²`, không phụ thuộc D. Nga gọi hàm này trong runner để tạo dòng tham chiếu đầu tiên của Bảng 1.

### 3.5 Test (`tests/test_local_optimizers.py`)
- Cả 3 lớp kế thừa `BaseOptimizer` và không định nghĩa lại `optimize` (kiểm tra `"optimize" not in cls.__dict__`).
- `actual_nfe ≤ nfe_budget` ở budget 200, 1000, 5000 với D = 1, 3, 28. Thêm kiểm tra riêng cho budget 2000 NFE của thí nghiệm B4 (D = 28).
- Coordinate Ascent: không chấp nhận bước hòa (đặt `J` hằng thì `x` không đổi); đúng `K−1` NFE mỗi tọa độ; dừng đúng điều kiện.
- Chạy 20 seeds × 3 D × 3 budgets trên `mock_fullshape` không lỗi.
- Tích hợp runner mà không sửa runner của Nga:

```bash
python run_benchmark.py --algos coordinate_ascent nelder_mead powell --budgets 200 --seeds 0-2
```

### 3.6 Tinh chỉnh siêu tham số (LD-7)
- Lưới `K ∈ {5, 7, 9}` × `w0 ∈ {0.3, 0.45}` có 6 tổ hợp, nhưng chỉ được thử **tối đa 5 cấu hình**. Bỏ đúng một tổ hợp và ghi lý do. Cấu hình mặc định `(K = 7, w0 = 0.45)` phải nằm trong 5 cấu hình.
- Seeds 100–104, chỉ dữ liệu Val. Cấu hình (D, λ, budget) dùng để tune theo đúng nội dung LD-7.
- Đóng băng cấu hình thắng vào `configs/hparams_frozen.yaml`.

### 3.7 Khảo sát địa hình (`src/core/landscape.py`, chỉ Val)

Dùng `λ = 0` cho B1 đến B4. Thêm một phân tích đối chiếu `λ = 0.1`: vẽ `J_{0.1}(t)` chồng lên `J_0(t)` trong B1 (quy ước đề xuất). Định nghĩa dùng trong báo cáo:
- **Bước nhảy:** điểm có `ΔJ ≠ 0` giữa hai điểm lưới kề nhau.
- **Plateau:** đoạn cực đại gồm ít nhất 2 điểm lưới liên tiếp có cùng `J`.
- **Cực trị địa phương:** do hàm hằng từng khúc, là điểm dừng của `coordinate_ascent` ở bước nhỏ nhất `w_min`; hai điểm cùng một plateau coi là một cực trị.

| # | Phân tích | Chi tiết bắt buộc |
|---|---|---|
| B1 | Đường cong 1D | `t ∈ [0.05, 0.95]`, bước 0.001 (901 điểm). Vẽ `J(t)` toàn cục và `F1_c(t)` cho 4 nhãn đại diện (1 Head, 1 Mid, 1 Tail, Neutral). Đánh dấu `t_g` và 0.5. Đếm số bước nhảy và số plateau. Lưu `landscape_1d_global.png` và `landscape_1d_per_label.png`. |
| B2 | Lát cắt 2D `(t_head, t_tail)` | Lưới 91×91 (bước 0.01), `t_mid = t_g`. Heatmap và contour. Đánh dấu cực đại lưới và các cực đại cục bộ. Lưu `landscape_2d_head_tail.png`. |
| B3 | Tỷ lệ phẳng | 10.000 điểm ngẫu nhiên đều trong `Θ`. Với mỗi điểm cộng nhiễu `U(−0.005, 0.005)` cho mỗi tọa độ (kẹp biên). Tính `P(ΔJ == 0 chính xác)` cho D = 1, 3, 28. Báo cáo kèm khoảng tin cậy Wilson 95%. Lưu `plateau_ratio_bar.png`. |
| B4 | Đếm cực trị địa phương | `coordinate_ascent` từ **200** điểm đầu ngẫu nhiên (seeds 200–399, tách khỏi seeds benchmark). D = 3: mỗi lần 1000 NFE. **D = 28: mỗi lần 2000 NFE** (ngân sách riêng Ngọc đã duyệt cho thí nghiệm này). |

**Lý do ngân sách 2000 NFE ở D = 28:** để cửa sổ kịp co về `w_min`, thuật toán cần ít nhất 10 lượt quét × 168 = 1680 NFE cộng 1 NFE cho `x0`. Với 2000 NFE tối đa hoàn thành 11 lượt quét. Ngân sách riêng này không ảnh hưởng benchmark chính thức: Bảng 1 vẫn giữ 3 budget 200, 1000, 5000 NFE.

Năm số liệu cần báo cáo cho B4 (mỗi D một bộ):
1. Số giá trị `J` cuối khác nhau (làm tròn `1e-4`).
2. Số cụm nghiệm bằng single-linkage, ngưỡng `L∞ = 0.02`. Dùng `scipy.cluster.hierarchy.linkage(method="single", metric="chebyshev")` và `fcluster(t=0.02, criterion="distance")`.
3. Tỷ lệ lần chạy đạt trong 0.001 so với lần tốt nhất.
4. Tỷ lệ lần chạy dừng bằng điều kiện hội tụ (`is_converged=True`) so với dừng do hết ngân sách.
5. Histogram `J` cuối (`landscape_local_optima_hist.png`).

Kết luận phải có **số cụ thể**, không dùng từ "nhiều".

### 3.8 Bảng và báo cáo
- `outputs/tables/landscape_summary.csv` gồm các cột `analysis_id, metric, D, lam, nfe_budget, value, unit, source_script`. Bảng phải có: số bước nhảy 1D, số plateau, plateau ratio D = 1/3/28, số giá trị `J` khác nhau, số cụm cực trị, tỷ lệ đạt trong 0.001 và tỷ lệ hội tụ (cho D = 3 và D = 28).
- `reports/landscape_analysis.md` (tối đa 2 trang): mô tả, số liệu, kết luận có số. Mọi con số dẫn từ CSV, không gõ tay.

### 3.9 Cổng G2 và Pull Request
- Chạy thử nội bộ 3 thuật toán trên 3 budget (200, 1000, 5000).
- Tạo PR để Ngọc chạy `scripts/audit_nfe.py` độc lập.
- Danh sách tự kiểm tra trước khi tạo PR:
  - Chữ ký `expand_theta`, `predict_labels`, `make_objective` khớp 100% mục 2.4 và 2.5.
  - 3 thuật toán chỉ cài `_search`.
  - Mọi đánh giá đi qua `tracker(theta)`.
  - Lệnh grep của CI sạch.
  - `pytest tests/test_decision.py tests/test_objective.py tests/test_local_optimizers.py` xanh 100%.
  - Có log đo thời gian hàm mục tiêu.
  - Mọi số liệu trong báo cáo dẫn từ CSV.

### 3.10 Xử lý sự cố
- **Lọt NFE (G2 thất bại):** SciPy không dừng ngay khi chạm `maxfev`. Thy không dựa vào `maxfev`, để `BudgetExhausted` từ `NFETracker` tự ném ra và được `BaseOptimizer.optimize()` bắt.
- **Trôi trên plateau:** nguyên nhân là chấp nhận `J_new ≥ J_cur`. Khóa điều kiện tăng chặt: nếu bằng nhau thì giữ nguyên tọa độ cũ.

## 4. Output tuần

**Mã nguồn:** `src/optimizers/local/coordinate_ascent.py`, `nelder_mead.py`, `powell.py`, hàm `evaluate_constant`, `src/core/landscape.py`.

**Test:** `tests/test_local_optimizers.py`.

**Cấu hình:** `configs/hparams_frozen.yaml`.

**Bảng:** `outputs/tables/landscape_summary.csv`.

**Biểu đồ:** `landscape_1d_global.png`, `landscape_1d_per_label.png`, `landscape_2d_head_tail.png`, `landscape_local_optima_hist.png`, `plateau_ratio_bar.png`.

**Báo cáo:** `reports/landscape_analysis.md`.

## 5. Definition of Done (DoD Tuần 2)

- [ ] 3 thuật toán kế thừa `BaseOptimizer`, chỉ cài `_search`; chạy 20 seeds × 3 D × 3 budgets trên `mock_fullshape` không lỗi.
- [ ] `actual_nfe ≤ nfe_budget` ở mọi dòng, kể cả các lần chạy B4 (1000 NFE ở D = 3, 2000 NFE ở D = 28).
- [ ] Qua Cổng G2: `audit_nfe.py` xác nhận chênh lệch bằng **0** tuyệt đối ở cả 3 thuật toán.
- [ ] `landscape_summary.csv` có đủ số liệu B1 đến B4; báo cáo có kết luận kèm số cụ thể.
- [ ] Siêu tham số: tối đa 5 cấu hình, seeds 100–104, đã đóng băng trong `hparams_frozen.yaml`.
- [ ] `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` không trả về kết quả nào.
- [ ] PR được Ngọc duyệt và merge vào `main`.