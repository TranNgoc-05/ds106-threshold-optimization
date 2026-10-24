# TASK_03_Vinh_Week1 — Giải tích Soft Macro-F1 & Khung 5-Fold CV

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh EVAL: toán học giải tích khả vi và đánh giá sau tối ưu (DS106 — ViGoEmotions) |
| **Owner** | Vinh |
| **Reviewers** | Trần Khánh Ngọc (Leader, A: nghiệm thu), Thy (Peer: đối chiếu tính tái lập) |
| **Deadline** | Cuối Tuần 1, tương ứng mốc M1 (surrogate + gradcheck nháp trên mock) |
| **Phối hợp** | Ngọc (C: kiểm tra công thức, slide báo cáo sơ bộ), Thy (C: mock data, `cv_folds_mock.npy`) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Chốt công thức giải tích của hàm xấp xỉ trơn Soft Macro-F1 `S_τ(θ)` và đạo hàm giải tích chính xác `∂S_τ/∂θ_g`, kể cả cộng gộp gradient theo nhóm khi `D < 28`.
2. Cài đặt `make_surrogate(P, Y, mapping_mode, lam, t_g)` trong `src/optimizers/gradient/surrogate.py`, chạy thông suốt trên mock.
3. Chạy **gradcheck nháp** trên `mock_fullshape/` để bắt lỗi đạo hàm sớm. Gradcheck chính thức nằm ở Tuần 2.
4. Soạn `reports/surrogate_notes.md` nêu rõ nguồn sai lệch cấu trúc giữa surrogate và `J_λ` gốc.
5. Dựng khung 5-Fold Multilabel Stratified CV (`make_folds`) và chứng minh khớp bit với `cv_folds_mock.npy` do Thy sinh.
6. Rà soát công thức cùng Ngọc cho slide báo cáo tiến độ Tuần 1.

---

## 2. Input đầu vào

**Từ các thành viên khác**

- Thy: `02_SAMPLE_DATA/mock_fullshape/` (đủ 28 nhãn, dùng cho gradcheck nháp), `02_SAMPLE_DATA/mock_small/` (smoke test shape/dtype), `02_SAMPLE_DATA/cv_folds_mock.npy` (sinh từ Tuần 1).
- Ngọc: hợp đồng kỹ thuật `src/contract/constants.py` (có `expand_theta`, `LO=0.05`, `HI=0.95`), `nfe_tracker.py`, `base_optimizer.py`.
- Giáo trình DS106: Chapter 3 *Optimization Algorithm Selection* (pp. 1–15), Chapter 2 *Optimization and Machine Learning* (pp. 1–18).
- Sechidis et al. (ECML PKDD 2011), *On the Stratification of Multi-label Data*.

**Quy tắc kỹ thuật áp dụng cả tuần**

- **LD-2:** 1 lần `evaluate_surrogate` = 1 NFE, 1 lần đánh giá rời rạc = 1 NFE, cả hai đều trừ vào ngân sách `NFETracker`. Tuần này chưa chạy tracker nhưng `surrogate.py` phải thuần hàm để tracker bọc được ở Tuần 2.
- **Cách ly Test:** tuần này không mở, không import `P_test`, `Y_test` ở bất kỳ module nào. Chỉ dùng `P_val`, `Y_val`, mock.
- **K-3:** số hạng phạt L2 là **regularization** kéo nghiệm về `t_g` để chống quá khớp. Coupling giữa các nhãn đến hoàn toàn từ Neutral độc quyền và Fallback. Không viết phạt L2 như một nguồn coupling ở bất kỳ tài liệu nào.
- Không có CafeBERT, không có `transfer.py`. Chỉ dùng xác suất ViSoBERT.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Thiết lập mô hình toán học Soft Macro-F1

Với nhiệt độ `τ > 0`, ngưỡng từng nhãn `t = expand_theta(θ) ∈ R^28`:

```text
s_ic   = σ((P_ic − t_c) / τ) = 1 / (1 + exp(−(P_ic − t_c)/τ))      # scipy.special.expit
n_c    = Σ_i Y_ic                                                 # số mẫu dương thực tế của nhãn c
A_c    = Σ_i Y_ic · s_ic
B_c    = Σ_i s_ic + n_c + ε,          ε = 1e-8
F̃_c   = 2 · A_c / B_c                 # tương đương 2TP/(2TP+FP+FN) với TP, FP, FN mềm
S_τ(θ) = (1/28) Σ_{c=1..28} F̃_c  −  λ · (1/D) · ‖θ − t_g·1‖²
```

Yêu cầu:

- Dùng `scipy.special.expit`, không tự cài `1/(1+exp(-x))` (tránh tràn số float).
- Ép `P`, `Y` về `float64` bên trong `make_surrogate` để tránh sai khác độ chính xác khi so với sai phân số học.
- Chế độ `global` (D=1): đặt `λ_eff = 0` (theo quy định của task). Các chế độ còn lại dùng `λ` truyền vào.
- Chế độ `semantic4` (D=4) cũng phải được hỗ trợ vì Bảng 2 và Ablation dùng D=4.

### 3.2. Khai triển đạo hàm giải tích

Đặt `s'_ic = ∂s_ic/∂t_c = −s_ic(1 − s_ic)/τ`.

```text
∂F̃_c/∂t_c = 2 · ( (Σ_i Y_ic·s'_ic)·B_c − A_c·(Σ_i s'_ic) ) / B_c²
```

Quy tắc chuỗi qua `expand_theta`: với `t = M·θ` (ma trận ánh xạ nhóm `M ∈ {0,1}^{28×D}`, mỗi hàng đúng một số 1), ta có `∇_θ = Mᵀ ∇_t`. Tức là gradient của `θ_g` là **tổng** gradient theo `t_c` của các nhãn `c` thuộc nhóm `g`:

```text
∂S_τ/∂θ_g = (1/28) · Σ_{c ∈ nhóm g} ∂F̃_c/∂t_c  −  2λ(θ_g − t_g)/D
```

- `per_label` (D=28): mỗi nhóm có đúng một nhãn, không cộng gộp.
- `D < 28`: cộng gộp theo nhóm. Quên bước này là nguyên nhân lỗi gradcheck phổ biến nhất (xem sổ tay sự cố của Word).
- Kiểm tra trên giấy một trường hợp nhỏ (N=3 mẫu, 2 nhãn cùng nhóm) trước khi viết code, và đối soát ma trận với Ngọc.

### 3.3. Đóng gói `make_surrogate` — `src/optimizers/gradient/surrogate.py`

Chữ ký bị khóa:

```python
def make_surrogate(P, Y, mapping_mode, lam, t_g) -> "Callable[[np.ndarray, float], Tuple[float, np.ndarray]]":
    """Trả về g(theta, tau) -> (S_tau(theta), grad_S_tau(theta)); S càng lớn càng tốt (ascent)."""
```

Cài đặt tham chiếu:

```python
import numpy as np
from scipy.special import expit
from src.core.decision import expand_theta   
# đối chiếu đường dẫn/chữ ký thực tế trong contract

EPS = 1e-8
N_LABELS = 28
_D_BY_MODE = {"global": 1, "freq3": 3, "semantic4": 4, "per_label": 28}


def _group_matrix(mapping_mode):
    D = _D_BY_MODE[mapping_mode]
    # Dựng M (28 x D) bằng cách áp expand_theta lên các vector cơ sở
    M = np.stack([np.asarray(expand_theta(e, mapping_mode), dtype=np.float64)
                  for e in np.eye(D)], axis=1)
    assert M.shape == (N_LABELS, D)
    assert set(np.unique(M)) <= {0.0, 1.0} and np.allclose(M.sum(axis=1), 1.0), \
        "expand_theta phải là phép gán nhóm 0/1, mỗi nhãn thuộc đúng một nhóm"
    return M, D


def make_surrogate(P, Y, mapping_mode, lam, t_g):
    P = np.asarray(P, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    M, D = _group_matrix(mapping_mode)
    n_c = Y.sum(axis=0)
    lam_eff = 0.0 if mapping_mode == "global" else float(lam)

    def g(theta, tau):
        theta = np.asarray(theta, dtype=np.float64)
        t = M @ theta                                   # (28,)
        s = expit((P - t[None, :]) / tau)               # (N, 28)
        ds = -s * (1.0 - s) / tau
        A = (Y * s).sum(axis=0)
        B = s.sum(axis=0) + n_c + EPS
        F = 2.0 * A / B
        dA = (Y * ds).sum(axis=0)
        dB = ds.sum(axis=0)
        dF_dt = 2.0 * (dA * B - A * dB) / B ** 2        # (28,)
        diff = theta - t_g
        S = F.mean() - lam_eff * np.mean(diff ** 2)
        grad = (M.T @ dF_dt) / N_LABELS - 2.0 * lam_eff * diff / D
        return float(S), grad

    return g
```

Tiêu chí kiểm tra:

- Trả về `(float, ndarray shape (D,))`, không NaN/Inf ở `τ ∈ {0.1, 0.03, 0.01}` trên `mock_small` và `mock_fullshape`.
- `P`, `Y` không bị sửa tại chỗ (không có side effect).
- Hàm không gọi tracker. Tracker bọc ngoài ở Tuần 2.

### 3.4. Gradcheck nháp trên mock (bắt lỗi đạo hàm sớm)

Tạo `tests/test_surrogate_gradcheck.py` ở trạng thái **nháp**, chạy trên `mock_fullshape/`:

```text
Sai phân trung tâm:  ∇_j^num S_τ(θ) = ( S_τ(θ + h·e_j) − S_τ(θ − h·e_j) ) / (2h),   h = 1e-6
RelErr = ‖∇_analytic − ∇_num‖₂ / max(‖∇_num‖₂, 1e-12)
```

- Phạm vi nháp: 5 điểm θ ngẫu nhiên × `τ ∈ {0.1, 0.01}` × 4 chế độ `{global, freq3, semantic4, per_label}`, seed cố định.
- θ lấy trong `[LO, HI] = [0.05, 0.95]`.
- Mục đích tuần này là **bắt lỗi sớm**. Mốc nghiệm thu chính thức (20 điểm × 3 τ × 3 chế độ, RelErr < 1e-5) nằm ở Tuần 2.
- Nếu RelErr ≥ 1e-5, rà theo thứ tự: (i) dấu trừ trong `s'`, (ii) cộng gộp theo nhóm, (iii) `h`.
- Thông tin tham khảo (không thuộc DoD): in `|S_τ(θ) − Macro-F1 rời rạc|` ở τ giảm dần trên mock để xác nhận surrogate tiến về hàm bậc thang khi τ → 0 (có phần sai lệch do Neutral/Fallback).

### 3.5. Tài liệu `reports/surrogate_notes.md`

Cấu trúc bắt buộc:

1. Định nghĩa `S_τ(θ)` và ký hiệu (đồng nhất với 3.1).
2. Dẫn xuất đạo hàm từng bước, gồm quy tắc thương và quy tắc chuỗi qua `expand_theta`.
3. Giải thích Macro-F1 rời rạc là hàm hằng từng khúc (piecewise constant) theo ngưỡng, nên đạo hàm hầu khắp nơi bằng 0 hoặc không xác định, vì sao cần surrogate.
4. **Nguồn sai lệch cấu trúc giữa nghiệm surrogate và `J_λ` gốc** (phục vụ H4):
   - (a) Surrogate **không mô hình hóa Neutral độc quyền và Fallback**, vì hai cơ chế này rời rạc và không khả vi.
   - (b) Tham số `τ` làm phẳng bậc thang, nên cực trị của `S_τ` lệch khỏi cực trị của `J_λ`.
5. Bản chất số hạng phạt (K-3): L2 là regularization kéo nghiệm về `t_g` nhằm chống quá khớp. Coupling giữa các nhãn đến từ Neutral độc quyền và Fallback, không phải từ L2.
6. Hạn chế đã biết và liên hệ H4.

### 3.6. Khung 5-Fold Multilabel Stratified CV — `src/eval/cv_lambda.py` (bản dự thảo)

Vinh **viết** `make_folds`. Thy **sinh** `02_SAMPLE_DATA/cv_folds_mock.npy` từ Tuần 1, Vinh kiểm tra khớp bit.

```python
import numpy as np

def make_folds(Y, n_splits=5, seed=42):
    """Trả về mảng int8 shape (N,), giá trị fold 0..n_splits-1."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(Y)), Y)
    except ImportError:                       # chỉ khi không cài được iterstrat
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(Y)))
    folds = np.full(len(Y), -1, dtype=np.int8)
    for f, (_, te) in enumerate(split_iter):
        folds[te] = f
    assert (folds >= 0).all()
    return folds
```

- Cố định `random_state=42`, `shuffle=True`, `n_splits=5`. Ghim phiên bản `iterstrat` trong `requirements` và ghi vào báo cáo, vì khớp bit phụ thuộc phiên bản.
- Nếu phải dùng `KFold` dự phòng thì ghi rõ trong báo cáo và báo Thy dùng cùng cách.
- CLI dự thảo: `python -m src.eval.cv_lambda --mock` chỉ chia fold và kiểm tra, **chưa** chạy CV (CV thật ở Tuần 2).

**Test khớp bit trên mock** (`tests/test_cv_folds_mock.py`):

```python
folds_vinh = make_folds(Y_mock, 5, 42)
folds_thy = np.load("02_SAMPLE_DATA/cv_folds_mock.npy")
assert folds_vinh.dtype == folds_thy.dtype == np.int8
assert folds_vinh.shape == folds_thy.shape
assert np.array_equal(folds_vinh, folds_thy)      # khớp từng bit
```

**Thống kê phân tầng** (in ra, dùng cho DoD): với mỗi nhãn, số mẫu dương mỗi fold, tỷ lệ dương mỗi fold, độ lệch chuẩn giữa 5 fold, đặc biệt cho các nhãn Tail. Lưu ý mock là dữ liệu giả nên con số chỉ mang tính chỉ báo. Kiểm tra lại trên `Y_val` thật chỉ để xem phân tầng, không tối ưu gì.

### 3.7. Phối hợp báo cáo sơ bộ Tuần 1 cùng Ngọc

- Rà soát công thức hàm mục tiêu rời rạc `J_λ`, số hạng phạt L2 (đúng theo K-3) và giải thích tính không khả vi (piecewise constant) của Macro-F1.
- Đối soát ma trận gradient `Mᵀ ∇_t` với Ngọc trên ví dụ giấy ở 3.2.
- Chuyển cho Ngọc đoạn tóm tắt 1 trang cho slide: công thức `S_τ`, `∇S_τ`, nguồn sai lệch cấu trúc.

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Mã nguồn | `src/optimizers/gradient/surrogate.py` |
| Mã nguồn | `src/eval/cv_lambda.py` (bản dự thảo: `make_folds`, CLI `--mock`) |
| Test | `tests/test_surrogate_gradcheck.py` (bản nháp), `tests/test_cv_folds_mock.py` |
| Tài liệu | `reports/surrogate_notes.md` |
| Dữ liệu | `02_SAMPLE_DATA/cv_folds_mock.npy` (do Thy sinh, Vinh kiểm tra khớp bit) |
| Ghi chú | Đoạn tóm tắt công thức cho slide báo cáo sơ bộ (gửi Ngọc) |

---

## 5. Definition of Done (DoD tuần)

- [ ] Công thức đạo hàm giải tích được kiểm tra trên giấy và đối soát ma trận với Ngọc.
- [ ] `make_surrogate` chạy thông suốt trên `mock_small` và `mock_fullshape`, đủ 4 chế độ mapping, không lỗi shape/dtype, không NaN/Inf ở `τ ∈ {0.1, 0.03, 0.01}`.
- [ ] Gradcheck nháp trên mock cho RelErr < 1e-5 (hoặc có ghi chú lỗi đã biết và kế hoạch sửa trước Tuần 2).
- [ ] `reports/surrogate_notes.md` hoàn chỉnh, nêu rõ cả hai nguồn sai lệch (Neutral/Fallback không mô hình hóa, τ làm phẳng) và bản chất L2 theo K-3.
- [ ] `make_folds` chia đúng 5 fold, khớp bit với `cv_folds_mock.npy` của Thy (`np.array_equal`, dtype `int8`, shape đúng).
- [ ] Độ lệch chuẩn tỷ lệ dương giữa 5 fold của các nhãn Tail được báo cáo và nhỏ. Mọi nhãn có mẫu dương ở mọi fold.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py` (hoặc rỗng nếu gate chưa tồn tại).
- [ ] PR gửi Ngọc review, không vi phạm LD-2 và không đụng tập Test.