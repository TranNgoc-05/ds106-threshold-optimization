# TASK_03_Vinh_Week2 — Gradcheck chính thức, Adam / L-BFGS-B, Chọn λ* bằng 5-Fold CV, Cổng G2/G3

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh EVAL: thuật toán gradient trên surrogate và chọn hệ số phạt λ* |
| **Owner** | Vinh |
| **Reviewers** | Trần Khánh Ngọc (Leader, A: nghiệm thu, kiểm toán G2, ký G3), Thy (Peer: cung cấp thuật toán CA, đối chiếu tái lập) |
| **Deadline** | Cuối Tuần 2, tương ứng mốc M2 (Adam/L-BFGS-B qua smoke test) và M3 (`lambda_star.json`) |
| **Phối hợp** | Ngọc (Cổng G2: audit NFE; Cổng G3: đóng băng `hparams_frozen.yaml`), Thy (`coordinate_ascent`, `compute_t_g`, `make_objective`) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Gradcheck **chính thức**: RelErr < 1e-5 trên 20 điểm × 3 mức τ × 3 chế độ mapping.
2. Hoàn thiện `adam_surrogate` và `lbfgsb_surrogate` với lịch ủ nhiệt 5 giai đoạn, tuân thủ LD-2.
3. Chạy 5-Fold CV **thật** trên tập Validation để chọn `λ*` cho D=3 và D=28 theo quy tắc 1-SE.
4. Nghiệm thu Cổng G2 (audit NFE chênh lệch = 0) và Cổng G3 (đóng băng λ* và siêu tham số) cùng Ngọc.

---

## 2. Input đầu vào

- Thy: `P_val.npy`, `Y_val.npy`, `artifacts/t_g.json`, `compute_t_g`, `coordinate_ascent.py`, `make_objective`.
- Ngọc: `BaseOptimizer`, `NFETracker` (có `evaluate_surrogate`, `BudgetExhausted`), `audit_nfe.py`.
- Vinh (Tuần 1): `surrogate.py`, `make_folds`, `surrogate_notes.md`.
- **Commit neo:** CV Tuần 2 phải chạy trên **một commit hash cố định của Thy chốt ở cuối Tuần 1** (code `compute_t_g`, `coordinate_ascent`, `make_objective`). Ghi hash đó vào `reports/cv_lambda_report.md` và không đổi giữa chừng.
- Đọc trước: Kingma & Ba (ICLR 2015) *Adam*; Byrd, Lu, Nocedal & Zhu (1995) *L-BFGS-B*; Hastie, Tibshirani & Friedman (2009) *ESL* (quy tắc 1-SE).

**Quy tắc kỹ thuật áp dụng cả tuần**

- **LD-2:** 1 `evaluate_surrogate` = 1 NFE, 1 đánh giá rời rạc `tracker(θ)` = 1 NFE.
- **Best luôn là điểm rời rạc:** `best_score`, `best_theta` chỉ cập nhật từ `J_λ` rời rạc, không lấy từ `S_τ`.
- **Cách ly Test:** CV và tuning chỉ dùng Validation. Không import/mở `P_test`, `Y_test`.
- **K-3:** L2 là regularization về `t_g` chống quá khớp. Coupling đến từ Neutral độc quyền và Fallback.
- Cả hai thuật toán kế thừa `BaseOptimizer`, chỉ cài `_search`, **không ghi đè** `optimize()`.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Gradcheck chính thức — `tests/test_surrogate_gradcheck.py`

Mở rộng bản nháp Tuần 1 thành bản chính thức:

```text
∇_j^num S_τ(θ) = ( S_τ(θ + h·e_j) − S_τ(θ − h·e_j) ) / (2h),    h = 1e-6
RelErr = ‖∇_analytic − ∇_num‖₂ / max(‖∇_num‖₂, 1e-12)   <   1e-5
```

- 20 điểm θ ngẫu nhiên độc lập (seed cố định) trong `[0.05, 0.95]^D`.
- `τ ∈ {0.1, 0.03, 0.01}`.
- Ba chế độ bắt buộc `{global (D=1), freq3 (D=3), per_label (D=28)}`, cộng `semantic4 (D=4)` như kiểm chứng bổ sung cho cấu hình D=4 ở Tuần 4 (không thay đổi điều kiện bắt buộc).
- Với `freq3`, `semantic4`, `per_label` dùng `λ = 0.1` (và thêm một lượt `λ = 0` để cô lập thành phần phạt). Với `global` dùng `λ = 0` (theo quy định của task).
- Dữ liệu: `mock_fullshape/` và một lượt trên `P_val`, `Y_val` thật.
- Ghi log vào `03_SUBMISSIONS/Vinh/test_logs/gradcheck_output.txt` (RelErr lớn nhất theo từng tổ hợp).

```python
import numpy as np, pytest
from src.optimizers.gradient.surrogate import make_surrogate

H = 1e-6
@pytest.mark.parametrize("mode,D", [("global",1), ("freq3",3), ("semantic4",4), ("per_label",28)])
@pytest.mark.parametrize("tau", [0.1, 0.03, 0.01])
def test_gradcheck(mode, D, tau, P, Y, t_g):
    lam = 0.0 if mode == "global" else 0.1
    g = make_surrogate(P, Y, mode, lam, t_g)
    rng = np.random.default_rng(2024)
    for _ in range(20):
        theta = rng.uniform(0.05, 0.95, size=D)
        _, ga = g(theta, tau)
        gn = np.empty(D)
        for j in range(D):
            e = np.zeros(D); e[j] = H
            gn[j] = (g(theta + e, tau)[0] - g(theta - e, tau)[0]) / (2 * H)
        rel = np.linalg.norm(ga - gn) / max(np.linalg.norm(gn), 1e-12)
        assert rel < 1e-5, (mode, tau, rel)
```

### 3.2. Lịch ủ nhiệt độ (dùng chung cho 2 thuật toán)

```text
τ_k = 0.1 · (0.1)^(k/4),  k = 0..4   ⟹   τ ∈ {0.1, 0.0562, 0.0316, 0.0178, 0.01}
```

- Phân bổ NFE: sau lệnh đánh giá `x0` (1 NFE), `remaining = max_nfe − 1`. Giai đoạn `k` nhận `⌊remaining_k / (5 − k)⌋` NFE, với `remaining_k` là NFE còn lại tại thời điểm vào giai đoạn.
- Mỗi giai đoạn gồm cả NFE gradient và NFE đánh giá rời rạc, không vượt ngân sách giai đoạn.
- Ghi `extra["tau_schedule"]` (5 giá trị τ) và `extra["stage_end_nfe"]` (NFE tích lũy cuối từng giai đoạn).

### 3.3. `adam_surrogate.py`

Đặc tả (cực đại hóa `S_τ`, ascent):

```text
đầu thuật toán: tracker(x0)                                  # 1 NFE rời rạc, luôn có best_theta
β1 = 0.9, β2 = 0.999, ε = 1e-8
với k = 0..4:
    τ = τ_k ;  α = α0 · 0.5^k            (α0 mặc định 0.02)
    m = 0, v = 0, t_step = 0             # BẮT BUỘC reset moment ở đầu mỗi giai đoạn
    trong khi còn NFE giai đoạn (chừa 1 NFE cho lần đánh giá rời rạc cuối giai đoạn):
        S, g = tracker.evaluate_surrogate(θ, τ)               # 1 NFE
        t_step += 1
        m = β1·m + (1−β1)·g ;  v = β2·v + (1−β2)·g²
        m̂ = m/(1−β1^t_step) ;  v̂ = v/(1−β2^t_step)
        θ_mới = clip(θ + α · m̂/(√v̂ + ε), LO, HI)             # clip [0.05, 0.95]
        nếu k = 4: đếm số bước liên tiếp có ‖θ_mới − θ‖∞ < 1e-5
        θ = θ_mới
        mỗi eval_every bước (mặc định 10) nếu còn đủ ngân sách: tracker(θ)     # 1 NFE rời rạc
    cuối giai đoạn: tracker(θ)                                  # 1 NFE rời rạc cập nhật best
dừng sớm (trả True = is_converged) nếu ‖Δθ‖∞ < 1e-5 trong 5 bước liên tiếp ở giai đoạn cuối
```

- Ngân sách phải được tính chặt để mọi lần đánh giá đều đi qua `tracker` (không có "NFE lọt").
- Lý do reset moment: `τ` đổi làm độ lớn và hướng gradient đổi đột ngột, moment cũ mang lịch sử của mặt phẳng cũ gây phân kỳ hoặc kẹt biên.
- Không ghi đè `optimize()`.

### 3.4. `lbfgsb_surrogate.py`

Mỗi giai đoạn `τ_k`:

```python
from scipy.optimize import minimize
res = minimize(
    fun=lambda x: tuple(-v for v in tracker.evaluate_surrogate(x, tau_k)),   # (−S, −∇S)
    x0=theta, jac=True, method="L-BFGS-B",
    bounds=[(LO, HI)] * D,
    options={"maxfun": stage_budget - 1, "ftol": 1e-9, "gtol": 1e-7, "maxcor": maxcor},
)
theta = np.clip(res.x, LO, HI)
tracker(theta)            # 1 NFE rời rạc cuối giai đoạn, rồi sang τ kế tiếp bắt đầu từ θ này
```

- `fun` trả về `(−S, −∇S)`, vì SciPy tối thiểu hóa.
- **`BudgetExhausted`:** SciPy line search có thể cố gọi `fun` thêm khi chạm trần. Không bắt trong `fun`, để ngoại lệ lan ra và để `BaseOptimizer.optimize()` xử lý, nhờ đó `is_converged` được ghi nhận chính xác.
- `is_converged=True` với `actual_nfe` nhỏ là hiện tượng **cần ghi nhận làm dữ liệu cho H4**, không phải lỗi.
- Ghi `extra["tau_schedule"]`, `extra["stage_end_nfe"]`, và số vòng lặp SciPy mỗi giai đoạn.

### 3.5. Tuning siêu tham số (LD-7, ≤ 5 cấu hình mỗi thuật toán)

- **Adam:** thử tối đa 5 cấu hình trong lưới `α0 ∈ {0.01, 0.02, 0.05}` × `eval_every ∈ {5, 10}` (6 tổ hợp, bỏ 1). Luôn gồm cấu hình mặc định `(0.02, 10)`, ghi rõ tổ hợp bị loại và lý do.
- **L-BFGS-B:** `maxcor ∈ {10, 20}` × số giai đoạn `∈ {3, 5}` (4 cấu hình).
- Dữ liệu: **chỉ Validation**, seeds `100–104`, NFE = 1000, D ∈ {3, 28}, `λ = λ*` (sau khi có kết quả 3.6).
- Tiêu chí chọn: trung bình `best_val_score` trên 5 seeds, lấy trung bình hai giá trị D. Hòa thì chọn cấu hình gần mặc định hơn.
- Ghi toàn bộ log tuning (đủ cấu hình đã thử và điểm) để nộp Ngọc.
- Seeds `100–104` tách biệt với seeds benchmark `0–19`.

### 3.6. 5-Fold CV chọn λ* — `src/eval/cv_lambda.py` (bản thật)

1. **Chia fold:** `make_folds(Y_val, 5, 42)`, lưu `artifacts/cv_folds.npy` (`(2066,)`, `int8`). Nếu phải dùng `KFold` dự phòng thì ghi rõ trong báo cáo.
2. **Với mỗi fold `f ∈ {0..4}`:**
   - Tính lại `t_g^f` **chỉ trên 4 fold huấn luyện** bằng `compute_t_g` của Thy.
   - Dùng `coordinate_ascent` của Thy tối ưu θ trên 4 fold huấn luyện, `NFE = 1000`, seeds `100..104`, `D ∈ {3, 28}`, `λ ∈ {0.0, 0.1, 0.5}`. Hàm mục tiêu là `J_λ` (có phạt) tại `t_g^f`, dùng `make_objective` của Thy.
   - Đánh giá **Macro-F1 không phạt** của θ trên fold giữ lại, theo cùng quy tắc quyết định của cấu hình `full` (Neutral độc quyền + Fallback argmax).
   - Tổng số lượt chạy: 2 D × 3 λ × 5 fold × 5 seeds = 150.
3. **Tổng hợp:** với mỗi `(D, λ)`, trung bình theo seeds trước, rồi theo 5 fold. `mean = trung bình 5 fold`, `SE = std(ddof=1) / √5` giữa 5 fold. Ghi rõ lựa chọn `ddof=1` trong báo cáo.
4. **Quy tắc 1-SE (đã đăng ký):** với mỗi `D`, gọi `λ_best = argmax mean`; `λ*_D` là λ **lớn nhất** thỏa

   ```text
   mean(λ) ≥ mean(λ_best) − 1 · SE(λ_best)
   ```
5. **Xuất:**
   - `outputs/tables/cv_lambda_results.csv`: cột `D, lambda, mean, se, mean_per_fold_1..5, selected_flag`.
   - `outputs/figures/cv_lambda_curve.png`: đường cong mean ± SE theo λ cho từng D, đánh dấu `λ_best`, ngưỡng 1-SE và `λ*`.
   - `artifacts/lambda_star.json` dạng `{"3": x, "28": y}` (khóa chuỗi).
   - `reports/cv_lambda_report.md`: phương pháp, **commit hash neo của Thy**, phiên bản `iterstrat`, seeds, bảng kết quả, quyết định chọn λ* và cách áp dụng 1-SE. Báo cáo kết quả theo số liệu thực, không giả định trước λ* nào.
6. **Cách ly Test:** `cv_lambda.py` không import, không mở file Test.

### 3.7. Nghiệm thu Cổng G2 và G3 cùng Ngọc

- **Cổng G2:** `tests/test_gradient_optimizers.py` chạy 2 thuật toán × 3 budget `{200, 1000, 5000}` × 3 D `{1, 3, 28}`. Kiểm tra: `actual_nfe ≤ max_nfe` và bằng đúng số đếm của tracker, nếu chưa hội tụ thì dùng hết đúng `max_nfe`. Tính lại `J_λ(best_theta)` rời rạc phải bằng `best_score`. Ngọc chạy `audit_nfe.py` xác nhận chênh lệch = 0.
- **Cổng G3:** nộp `lambda_star.json`, `cv_lambda_results.csv`, log tuning cho Ngọc ký duyệt đóng băng `configs/hparams_frozen.yaml`.
- Sau khi Ngọc ký duyệt, merge nhánh gradient vào `main`.

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Mã nguồn | `src/optimizers/gradient/surrogate.py`, `adam_surrogate.py`, `lbfgsb_surrogate.py` |
| Mã nguồn | `src/eval/cv_lambda.py` (bản thật) |
| Test | `tests/test_surrogate_gradcheck.py`, `tests/test_gradient_optimizers.py` |
| Log | `03_SUBMISSIONS/Vinh/test_logs/gradcheck_output.txt`, log tuning siêu tham số |
| Artifact | `artifacts/cv_folds.npy`, `artifacts/lambda_star.json` |
| Bảng, biểu đồ | `outputs/tables/cv_lambda_results.csv`, `outputs/figures/cv_lambda_curve.png` |
| Báo cáo | `reports/cv_lambda_report.md` (kèm commit hash neo của Thy) |

---

## 5. Definition of Done (DoD tuần)

- [ ] Gradcheck đạt `RelErr < 1e-5` trên **toàn bộ** 20 điểm × 3 τ × 3 chế độ (kèm `semantic4`), log lưu đủ.
- [ ] `adam_surrogate` đã reset moment ở đầu mỗi giai đoạn, clip `[0.05, 0.95]`, đánh giá rời rạc `x0` ở lệnh đầu, đánh giá rời rạc mỗi 10 bước và cuối giai đoạn, dừng sớm đúng điều kiện.
- [ ] `lbfgsb_surrogate` xử lý `BudgetExhausted` đúng, đánh giá rời rạc cuối mỗi giai đoạn.
- [ ] Cả hai thuật toán vượt Cổng G2: không vượt ngân sách, `best_score` luôn là `J_λ` rời rạc, `audit_nfe.py` chênh lệch = 0.
- [ ] Tuning ≤ 5 cấu hình mỗi thuật toán, chỉ dùng Validation, log đầy đủ.
- [ ] 5-Fold CV hoàn tất 100% (150 lượt chạy), `lambda_star.json` đúng định dạng và đúng quy tắc 1-SE, `cv_lambda_results.csv` ghi mean và SE từng `(D, λ)`.
- [ ] `cv_lambda_report.md` ghi commit hash neo của Thy, phiên bản `iterstrat`, lựa chọn `ddof`.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py` (hoặc rỗng nếu gate chưa tồn tại).
- [ ] Ngọc ký duyệt G2 và G3 (đóng băng `configs/hparams_frozen.yaml`), nhánh gradient được merge vào `main`.