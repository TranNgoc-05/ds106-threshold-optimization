# TASK_03_Vinh_Week3 — Cổng cách ly Test, Separable Oracle & Optimality Gap (D=28)

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh EVAL: bảo vệ tập Test và kiểm chứng tối ưu toàn cục chính xác |
| **Owner** | Vinh |
| **Reviewers** | Trần Khánh Ngọc (Leader, A: thẩm định logic niêm phong), Thy (Peer: đối chiếu `make_objective` và tính tái lập) |
| **Deadline** | Cuối Tuần 3 (không gắn mốc M riêng; sản phẩm phục vụ Cổng G4 của Ngọc và mốc M5) |
| **Phối hợp** | Ngọc (A), Nga (C: chạy runner cấu hình `sep_check`), Thy (C: `make_objective`) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Xây dựng cổng cách ly tập Test `src/eval/test_gate.py`, từ chối mở khi thiếu hoặc sai chữ ký niêm phong.
2. Cài đặt bộ giải phân tích tách biến `src/eval/separable_oracle.py` cho cấu hình `sep_check`, cho nghiệm tối ưu toàn cục chính xác `J*`.
3. Đo `Optimality_Gap = J* − best_val_score` cho cả 10 thuật toán ở D=28 qua 20 seeds × 3 budget, cung cấp bằng chứng trực tiếp cho **H3**.
4. Chuẩn bị sẵn sàng mở cổng Test khi Ngọc ban hành `LOCK_FREEZE.json`.

---

## 2. Input đầu vào

- Nhánh `main` đã tích hợp đủ 10 thuật toán và `configs/hparams_frozen.yaml`.
- `P_val.npy`, `Y_val.npy`, `artifacts/t_g.json`, `artifacts/lambda_star.json` (λ* từ Tuần 2, dùng `λ*_28`).
- `make_objective` của Thy (đối chiếu hàm mục tiêu), runner của Nga với cấu hình `sep_check`.
- Đọc trước: giáo trình DS106 Chapter 5 *Local and Global Optimization* (pp. 1–12); Lipton, Elkan & Naryanaswamy (2014) *Optimal Thresholding of Classifiers to Maximize F1 Measure*.

**Quy tắc kỹ thuật áp dụng cả tuần**

- **Cách ly Test tuyệt đối:** `test_gate.py` là module **duy nhất** được import, mở hoặc nhắc tới `P_test`, `Y_test`, `*_test.npy`. Gate chỉ cung cấp `load_test()` (không thêm hàm công khai nào khác).
- Mọi module khác lấy dữ liệu Test **chỉ qua** `load_test()` và không được chứa các chuỗi `P_test`, `Y_test`, `_test.npy` trong mã nguồn `src/`.
- **Chữ ký niêm phong** chỉ băm đúng 2 file: `outputs/benchmark_results.csv` (1.801 dòng) và `outputs/benchmark_theta_log.jsonl`.
- Các lượt chạy bổ sung phải ghi ra **file riêng**, không ghi đè vào hai file đã băm.
- **K-3:** phạt L2 là regularization về `t_g`. Coupling quy cho Neutral độc quyền và Fallback. Chính vì vậy `sep_check` (tắt cả hai, giữ L2 với `t_g` cố định) tách được thành 28 bài toán 1 chiều độc lập.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Cổng cách ly Test — `src/eval/test_gate.py`

```python
def load_test(use_mock: bool = False) -> "Tuple[np.ndarray, np.ndarray]":
    """Trả về (P_test, Y_test). Chỉ hàm này được phép chạm dữ liệu Test."""
```

**Chế độ chính thức (`use_mock=False`):**

1. Kiểm tra tồn tại `artifacts/LOCK_FREEZE.json`. Thiếu → `PermissionError`.
2. Đọc `LOCK_FREEZE.json`, thiếu trường băm của bất kỳ file nào trong 2 file → `PermissionError`.
3. Tính SHA-256 **thực tế** của `outputs/benchmark_results.csv` và `outputs/benchmark_theta_log.jsonl`:
   - Đọc ở chế độ nhị phân (`rb`), theo khối (chunk), không qua bộ giải mã văn bản. Việc này tránh sai lệch CRLF/LF do trình soạn thảo.
4. So khớp với chuỗi băm trong `LOCK_FREEZE.json`. Sai lệch dù **1 ký tự** → `PermissionError`, không nạp dữ liệu.
5. Chỉ khi khớp cả hai file mới nạp và trả về dữ liệu Test.

**Chế độ thử nghiệm (`use_mock=True`):**

- Chỉ nạp từ `02_SAMPLE_DATA/mock_*/`, không cần `LOCK_FREEZE.json`, không bao giờ trỏ vào dữ liệu thật.
- CLI: `python -m src.eval.test_gate --use_mock` (in shape để kiểm tra). Các script khác (`generalization.py`, `ablation.py`) nhận cờ `--use_mock` và chuyển tiếp vào `load_test(use_mock=...)`.

Lưu ý thiết kế:

- Đường dẫn (`LOCK_FREEZE`, 2 file benchmark, thư mục dữ liệu) khai báo thành hằng số cấp module để unit test dùng `monkeypatch` trỏ sang thư mục tạm.
- Thông báo lỗi không lộ nội dung hay shape dữ liệu Test.
- **Cần xác nhận với Ngọc ngay đầu tuần:** schema của `LOCK_FREEZE.json` (tên trường băm). Vinh viết gate theo schema Ngọc ban hành. Nếu Ngọc chưa chốt, ghi giả định vào docstring và thông báo.
- Đối chiếu tên file thực tế trong `mock_fullshape/` với Thy trước khi viết nhánh mock.

### 3.2. Unit test cổng — `tests/test_test_gate.py`

Tất cả test dùng dữ liệu giả trong thư mục tạm (`tmp_path` + `monkeypatch`), **không** mở Test thật:

| Trường hợp | Kỳ vọng |
|---|---|
| Thiếu `LOCK_FREEZE.json` | `PermissionError` |
| `LOCK_FREEZE.json` thiếu trường băm | `PermissionError` |
| Sửa **1 byte** của bản sao `benchmark_results.csv` | `PermissionError` |
| Sửa 1 byte của bản sao `benchmark_theta_log.jsonl` | `PermissionError` |
| Niêm phong hợp lệ, hai băm khớp | Trả về `(P, Y)` đúng shape |
| `use_mock=True` | Chạy được không cần niêm phong, chỉ đọc từ thư mục mock |

Kiểm thử thực tế trên repo thật (khi chưa có niêm phong): `python -c "from src.eval.test_gate import load_test; load_test()"` phải ném `PermissionError`. Lưu log vào `03_SUBMISSIONS/Vinh/test_logs/test_gate_security_log.txt`.

### 3.3. Bộ giải phân tích tách biến — `src/eval/separable_oracle.py`

Trong `sep_check` (`fallback=none`, `excl=False`, giữ phạt L2 với `t_g` cố định):

```text
J_sep(θ) = (1/28) · Σ_c [ F1_c(t_c) − λ · (t_c − t_g)² ],      λ = λ*_28, mapping per_label (D=28)
```

Phạt L2 tách được thành tổng 28 số hạng 1 chiều vì `t_g` cố định. Nghiệm toàn cục chính xác đạt được bằng tìm kiếm 1 chiều theo từng nhãn.

**Cấu trúc bài toán 1 chiều.** Với nhãn `c`, sắp xếp các giá trị xác suất phân biệt `p_1 < p_2 < … < p_K`. Với quy tắc dự đoán dương khi `P_ic ≥ t`, `F1_c(t)` là **hàm hằng trên mỗi khoảng nửa mở `(p_{k-1}, p_k]`**. Trên mỗi khoảng, `−λ(t − t_g)²` đạt cực đại tại điểm trong khoảng gần `t_g` nhất.

> **Kiểm tra bắt buộc đầu tuần:** đối chiếu quy tắc dự đoán trong `make_objective` của Thy là `P ≥ t` hay `P > t`. Nếu là `>`, các khoảng đổi thành `[p_k, p_{k+1})` và tập ứng viên đối xứng lại. Cố định cùng dtype (`float64`) cho `P` ở oracle và ở `make_objective`.

**Tập ứng viên rời rạc** (đã chốt): `{ p_k, nextafter(p_k, +inf), t_g, 0.05, 0.95 }` với mọi `p_k`, lọc về `[LO, HI] = [0.05, 0.95]`.

Tính đầy đủ của tập ứng viên: trên mỗi khoảng `(p_{k-1}, p_k]` cắt với `[LO, HI]`, điểm gần `t_g` nhất là một trong ba loại:
- chính `t_g` (nếu `t_g` nằm trong khoảng),
- đầu mút phải `p_k` (nếu `t_g > p_k`),
- điểm float nhỏ nhất trong khoảng, `nextafter(p_{k-1}, +inf)` (nếu `t_g ≤ p_{k-1}`).

Các đầu mút bị cắt bởi miền `[0.05, 0.95]` là `0.05` và `0.95`. Vậy tập trên bao phủ mọi cực đại cục bộ.

Giao diện đề xuất (khớp chữ ký trong contract nếu đã có, nếu chưa thì Ngọc khóa):

```python
def separable_oracle(P, Y, lam, t_g, lo=0.05, hi=0.95):
    """Trả về dict: J_star (float), t_star (28,), j_per_label (28,)."""
```

Cài đặt: với mỗi nhãn, sắp xếp `P[:, c]`, dùng tổng tích lũy để tính `TP`, `FP`, `FN` cho mọi ứng viên bằng `searchsorted` (không dùng vòng lặp O(N·K)). Quy ước `F1 = 0` khi mẫu số bằng 0, **phải trùng** với `make_objective`.

**Ba tiêu chuẩn kiểm thử** (`tests/test_separable_oracle.py`):

1. **Brute-force trên tập ứng viên:** cài đặt độc lập, đơn giản (vòng lặp qua từng ứng viên, tính dự đoán và F1 bằng định nghĩa). Kết quả khớp bản vector hóa với sai khác tuyệt đối `≤ 1e-9` trên mọi nhãn.
2. **Khớp `make_objective` của Thy tại θ\*:** `J*` do oracle trả về phải bằng giá trị `make_objective` (cấu hình `sep_check`, `λ*_28`) tại `t_star`, sai khác `≤ 1e-9`. Việc này kiểm chứng oracle và hàm mục tiêu nhất quán.
3. **Lưới 0.0005 không bao giờ vượt oracle:** quét lưới `[0.05, 0.95]` bước `0.0005` (1801 điểm) cho từng nhãn (ít nhất 3 nhãn ngẫu nhiên theo Word, nên chạy cả 28 nhãn vì rẻ). Lưới là tập con của miền nên giá trị lưới `≤ J*_c + 1e-12`. Đây là kiểm chứng đúng hướng (lưới có thể kém hơn nhưng không được hơn).

### 3.4. Đo Optimality Gap ở D=28 (bằng chứng cho H3)

- Cấu hình: `sep_check`, mapping `per_label` (D=28), `λ = λ*_28`, `t_g` cố định.
- 10 thuật toán × NFE `{200, 1000, 5000}` × 20 seeds (`0–19`) = 600 lượt chạy, bằng runner của Nga.
- Lệnh minh họa (tên cờ đối chiếu theo README runner): 

  ```bash
  python run_benchmark.py --mapping per_label --lam <lambda_star_28> <cờ sep_check: fallback=none, excl=False> \
      --budgets 200 1000 5000 --seeds 0-19 --out outputs/sepcheck_results.csv
  ```
- Ghi kết quả ra **file riêng** `outputs/sepcheck_results.csv`, không ghi vào `benchmark_results.csv`.
- Tính `Optimality_Gap = J* − best_val_score` cho từng lượt chạy.
- **Kiểm tra toàn vẹn:** không thuật toán nào được có `Optimality_Gap < −1e-9`. Nếu có thì dừng, báo Ngọc ngay (lỗi tracker, objective hoặc oracle). Không chỉnh số liệu.
- Lưu ý khi diễn giải: ở `sep_check`, surrogate của Adam/L-BFGS-B khớp cấu trúc với hàm mục tiêu (cùng không có Neutral/Fallback), nên sai lệch còn lại của hai thuật toán này chủ yếu do `τ` làm phẳng. Ghi nhận cho H4.
- **Xuất:**
  - `outputs/tables/optimality_gap_D28.csv`: cột `algo, nfe_budget, seed, J_star, best_val_score, optimality_gap`, thêm bảng tóm tắt trung vị/IQR theo (algo, budget).
  - `outputs/figures/optimality_gap_D28_boxplot.png`: biểu đồ hộp, 3 panel theo budget, một hộp mỗi thuật toán (cân nhắc trục `symlog` nếu gap trải rộng).
- Chỉ trình bày số liệu và biểu đồ, **chưa** kết luận H3. Kết luận nằm ở `eval_findings.md` (Tuần 4).
- `outputs/sepcheck_results.csv` được tái sử dụng cho Ablation Tuần 4 (cấu hình `sep_check`, `coordinate_ascent` và `differential_evolution` ở NFE=5000, seeds 0–19 trùng khớp).

### 3.5. Chuẩn bị cho Tuần 4 (không thuộc DoD)

- Dựng khung `generalization.py`, `ablation.py`, `stats_mwu_holm.py` chạy được với `--use_mock` trên dữ liệu giả.
- Dựng sớm `tests/test_stats.py` trên dữ liệu giả để rút ngắn Tuần 4.
- Đề nghị Ngọc tính SHA-256 trên môi trường Linux/CI và không mở lại CSV bằng Excel (nguyên nhân phổ biến gây `PermissionError` do CRLF/LF).

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Mã nguồn | `src/eval/test_gate.py`, `src/eval/separable_oracle.py` |
| Test | `tests/test_test_gate.py`, `tests/test_separable_oracle.py` |
| Log | `03_SUBMISSIONS/Vinh/test_logs/test_gate_security_log.txt` |
| Dữ liệu chạy | `outputs/sepcheck_results.csv` (file riêng, không nằm trong 2 file băm) |
| Bảng | `outputs/tables/optimality_gap_D28.csv` |
| Biểu đồ | `outputs/figures/optimality_gap_D28_boxplot.png` |

---

## 5. Definition of Done (DoD tuần)

- [ ] `test_gate.py` chỉ cung cấp `load_test()`. Từ chối mở khi thiếu `LOCK_FREEZE.json`, khi băm sai lệch (sửa 1 byte CSV hoặc JSONL), khi thiếu trường băm. Chạy được với `use_mock=True` và cờ `--use_mock`.
- [ ] Log bảo mật chứng minh cổng từ chối trên repo thật khi chưa có niêm phong hợp lệ.
- [ ] `separable_oracle` vượt cả 3 tiêu chuẩn: brute-force ứng viên ≤ 1e-9, khớp `make_objective` tại θ\* ≤ 1e-9, lưới 0.0005 không vượt oracle.
- [ ] 600 lượt chạy `sep_check` hoàn tất. Toàn bộ thuật toán có `Optimality_Gap ≥ −1e-9`.
- [ ] `optimality_gap_D28.csv` và `optimality_gap_D28_boxplot.png` được xuất, số liệu giữ nguyên như đo được.
- [ ] Không có lượt chạy bổ sung nào ghi đè vào `benchmark_results.csv` hoặc `benchmark_theta_log.jsonl`.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py`.
- [ ] Ngọc thẩm định logic niêm phong. Không có CafeBERT hay `transfer.py` trong nhánh EVAL.