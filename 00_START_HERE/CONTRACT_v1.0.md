# DS106 — Gói phân công Sprint 1

**Đề tài:** Tối ưu hóa Vector Ngưỡng Quyết định Đa nhãn trên ViGoEmotions: Khảo sát Địa hình Hàm số, So sánh Hiệu năng Thuật toán và Kiểm chứng Định lý No Free Lunch  
**Môn học:** DS106 – Tối ưu hóa ứng dụng, UIT (GV: TS. Nguyễn-Tất B. Thiện)  
**Phiên bản hợp đồng:** `contract v1.0` — **bản ban hành chính thức** (Leader Ngọc đã chốt các quyết định ở mục "Nhật ký chốt quyết định"). Mọi thay đổi sau đây phải qua Change Request, xem mục 2.13.

## Nhật ký chốt quyết định của Leader (áp dụng vào bản ban hành)

| # | Quyết định | Tác động trong gói |
|---|---|---|
| K-1 | **Bỏ hoàn toàn CafeBERT.** Dự án chỉ dùng một mô hình duy nhất là ViSoBERT. | Xóa `P_*_cafebert.npy`, `transfer.py`, `transfer_cafebert.csv/.png`, cột CafeBERT của Bảng 2, rủi ro R-3, tài liệu CafeBERT. EVAL-01 còn: surrogate gradient, 5-fold CV chọn λ*, tổng quát hóa trên Test, ablation, MWU-Holm. |
| K-2 | **Sửa quy tắc Neutral độc quyền** theo so sánh xác suất (không còn "luôn loại Neutral"). | LD-11, mục 2.4, golden cases, differential test, mô tả ablation `no_excl_neutral`. |
| K-3 | **Chốt OI-1 = phương án (a).** Giữ nguyên `J_λ` ở LD-8. Phạt L2 là regularization kéo nghiệm về `t_g`; ghép cặp giữa các nhãn đến hoàn toàn từ Neutral độc quyền và Fallback. | Mục 0 (ghi chú OI-1), LEAD-01 A4, EVAL-01 F, bảng rủi ro. |
| K-4 | **Chapter 6 = "Premature Convergence" (pp. 1–9)**, TS. Nguyễn-Tất B. Thiện (2026). | Danh mục đọc, ghi chú mục 7. |
| K-5 | **Duyệt trước R-4:** TPE ở D=28, nếu pilot thấy một lần chạy > 10 phút thì đặt `multivariate=False` (mặc định). | GLOBAL-01 A5, bảng rủi ro R-4. |

**Quy ước thời gian:** `D0` là ngày kickoff. `D+n 23:30` là deadline cứng tính theo ngày lịch. Khi triển khai, thay `D0` bằng ngày thật.  
**Nguyên tắc nối pipeline:** mọi module chỉ giao tiếp qua file và interface trong gói này. Không module nào được tự đổi tên file, shape, thứ tự cột hay chữ ký hàm.  
**Nguyên tắc cách ly dữ liệu:** `P_test.npy`/`Y_test.npy` chỉ được mở qua một cổng duy nhất (mục 2.12). Vi phạm là lỗi chặn merge.

---

## 0. Quyết định phải khóa (Locked Decisions)

| ID | Quyết định | Lý do |
|---|---|---|
| LD-1 | Mọi optimizer **maximize** `J_λ(θ)`. Wrapper `-J` chỉ nằm trong code của optimizer, không lộ ra interface. | Tránh lệch dấu giữa các thành viên. |
| LD-2 | 1 NFE = 1 lần gọi hàm mục tiêu bất kỳ: `J_λ` rời rạc **hoặc** surrogate `S_τ` (kèm hay không kèm gradient). Không có API batch nào được bỏ qua bộ đếm. | Công bằng giữa nhóm hộp đen và nhóm gradient. |
| LD-3 | Mỗi `(algorithm, D, seed, budget)` là **một lần chạy độc lập**. Cấm cắt snapshot từ một lần chạy 5000 NFE để báo cáo 200/1000. | SA, DE, PSO, L-BFGS-B có lịch trình phụ thuộc budget. |
| LD-4 | Mỗi thuật toán chạy **một lần, không restart** trong benchmark chính. Dừng sớm vì tiêu chí hội tụ nội bộ ghi `is_converged=True`, `actual_nfe < nfe_budget`. Biến thể restart nếu có phải đặt tên riêng, ví dụ `nelder_mead_restart`. | Giữ nguyên hiện tượng hội tụ sớm để phân tích (Chương 6). |
| LD-5 | Điểm khởi tạo `x0 = init_theta(seed, D)` dùng chung cho mọi thuật toán cần điểm xuất phát (CA, NM, Powell, SA, Adam, L-BFGS-B). | Seed phản ánh độ nhạy với điểm xuất phát; cho phép so sánh cặp. |
| LD-6 | Seeds benchmark: `0..19`. Seeds tuning siêu tham số: `100..104`. Hai tập không giao nhau. | Chống rò rỉ tuning vào benchmark. |
| LD-7 | **Quy tắc công bằng siêu tham số:** mỗi thuật toán được thử tối đa **5 cấu hình**, tuning trên Val tại NFE=1000, D∈{3,28}, seeds 100–104. Cấu hình thắng được đóng băng vào `configs/hparams_frozen.yaml` trước khi chạy benchmark chính. | Không thuật toán nào được tune nhiều hơn. |
| LD-8 | `J_λ(θ) = MacroF1(Y, Ŷ(θ)) − λ · (1/D) · ‖θ − t_g·1‖₂²`. Với D=1 luôn dùng `λ=0` (vì `t_g` chính là nghiệm D=1). | Chuẩn hóa theo D để λ có cùng thang đo ở D=3 và D=28. |
| LD-9 | `t_g` được tính **một lần** bằng grid search 1 chiều trên Val (bước 0.001, 901 điểm, `λ=0`), lưu `artifacts/t_g.json`. Hòa điểm: chọn trung điểm plateau cực đại liên tục dài nhất; nếu hòa tiếp, chọn plateau gần 0.5 nhất. Grid này không tính vào NFE. | `t_g` là hằng số của bài toán, không phải kết quả của optimizer. |
| LD-10 | Cấu hình pipeline đầy đủ ("Full"): `fallback_strategy="argmax"`, `exclusive_neutral=True`, `λ=λ*`. Mọi cấu hình khác phải ghi rõ là biến thể ablation. | Tránh nhầm lẫn khi đọc Bảng 1–3. |
| LD-11 | Quy tắc Neutral độc quyền (so sánh xác suất). Khi `neutral ∈ S_i` và `|S_i|>1`, đặt `m_i = max P[i,c]` với `c ∈ S_i \ {neutral}`: **nếu `P[i,neutral] > m_i` (chặt) thì GIỮ Neutral và LOẠI mọi cảm xúc khác trong `S_i`; ngược lại (kể cả bằng nhau) LOẠI Neutral và giữ các cảm xúc còn lại.** Thứ tự áp dụng: ngưỡng → neutral độc quyền → fallback. | Khớp ViGoEmotions và đề xuất nhóm (Leader chốt K-2). Hòa xác suất thiên về giữ cảm xúc rõ ràng. |
| LD-12 | Tie-break của `argmax` là `np.argmax` (chỉ số nhỏ nhất). Tie-break của "best" trong tracker là strict `>` (giữ nghiệm gặp đầu tiên). | Kết quả tái lập từng bit. |
| LD-13 | Ma trận lưu `float64` (xác suất) và `uint8` (nhãn). Không dùng `float32`. | Tránh sai khác làm lật nhãn ở sát ngưỡng. |

> **OI-1 — ĐÃ CHỐT phương án (a) (K-3).** `J_λ` giữ nguyên như LD-8. Với `t_g` cố định, số hạng phạt L2 là hàm tách biến theo từng `t_c`, nên vai trò của nó là **chống quá khớp (regularization), kéo nghiệm về `t_g`**; nó **không** tạo ghép cặp. **Sự ghép cặp (coupling) giữa các nhãn đến hoàn toàn từ cơ chế Neutral độc quyền và Fallback.** Câu chữ bắt buộc trong báo cáo: phạt L2 được mô tả là regularization; mọi phát biểu về coupling/suy biến D=28 phải quy về Neutral độc quyền và Fallback. Vinh vẫn kiểm chứng thực nghiệm bằng `separable_oracle` (EVAL-01 mục E). Công thức `J_λ` ở LD-8 là dạng chuẩn hóa theo D; Ngọc đã xác nhận giữ nguyên.

---

## 1. Cấu trúc thư mục làm việc chuẩn

```text
ds106-threshold-opt/
├── 00_START_HERE/
│   ├── README.md                      # tóm tắt dự án, cách chạy, thứ tự đọc
│   ├── CONTRACT_v1.0.md               # bản sao mục 2 của file này (nguồn sự thật)
│   ├── SUBMISSION_RULES.md            # quy tắc nộp, PR checklist (mục 5)
│   ├── CHANGE_REQUESTS.md             # nhật ký thay đổi hợp đồng
│   ├── requirements.txt               # pin chính xác bằng pip freeze ở M0
│   └── env.example                    # OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, ...
├── 01_TASKS/
│   ├── Week1/
│   │   ├── TASK_W1_01_Ngoc.md
│   │   ├── TASK_W1_02_Thy.md
│   │   ├── TASK_W1_03_Vinh.md
│   │   └── TASK_W1_04_Nga.md
│   ├── Week2/
│   │   ├── TASK_W2_01_Ngoc.md
│   │   ├── TASK_W2_02_Thy.md
│   │   ├── TASK_W2_03_Vinh.md
│   │   └── TASK_W2_04_Nga.md
│   ├── Week3/
│   │   ├── TASK_W3_01_Ngoc.md
│   │   ├── TASK_W3_02_Thy.md
│   │   ├── TASK_W3_03_Vinh.md
│   │   └── TASK_W3_04_Nga.md
│   └── Week4/
│       ├── TASK_W4_01_Ngoc.md
│       ├── TASK_W4_02_Thy.md
│       ├── TASK_W4_03_Vinh.md
│       └── TASK_W4_04_Nga.md
├── 02_SAMPLE_DATA/
│   ├── make_mock_data.py              # sinh toàn bộ mock (Thy, CORE-01 A1)
│   ├── mock_small/                    # N=200, unit test < 1 giây
│   │   ├── P_val.npy  Y_val.npy  P_test.npy  Y_test.npy
│   │   └── meta.json
│   ├── mock_fullshape/                # N=2066, đo thời gian/NFE, smoke test
│   │   ├── P_val.npy  Y_val.npy  P_test.npy  Y_test.npy
│   │   └── meta.json
│   ├── mappings/                      # bản mock của 3 file mapping (cùng schema thật)
│   ├── golden_cases.json              # ca kiểm thử decision rule có đáp án cố định
│   ├── cv_folds_mock.npy
│   └── benchmark_results_example.csv  # 12 dòng minh họa schema
├── 03_SUBMISSIONS/
│   ├── Ngoc/   Nga/   Thy/   Vinh/    # cấu trúc con ở mục 5.2
└── src/                               # kho mã nguồn dùng chung (merge qua PR)
│   ├── contract/                      # Ngọc sở hữu: nfe_tracker.py, base_optimizer.py, constants.py
│   ├── core/                          # Nga: data_io.py, decision.py, objective.py, landscape.py
│   ├── optimizers/
|   │   ├── local/                     # Thy: coordinate_ascent.py, nelder_mead.py, powell.py
|   │   ├── global_/                   # Nga: random_search.py, simulated_annealing.py, de.py, pso.py, tpe.py
|   │  └── gradient/                  # Vinh: surrogate.py, adam_surrogate.py, lbfgsb_surrogate.py
│   ├── runner/                        # Nga: run_benchmark.py, registry.py
│   ├── eval/                          # Vinh: cv_lambda.py, test_gate.py, generalization.py,
|   │                                 #       ablation.py, stats_mwu_holm.py, separable_oracle.py
│   ├── configs/                       # hparams_frozen.yaml, mappings/*.json
│   ├── artifacts/                     # t_g.json, lambda_star.json, theta_frozen/*.json, LOCK_FREEZE.json
│   ├── data/                          # P_val.npy ... (git-ignored, có checksum trong meta.json)
│   ├── outputs/                       # benchmark_results.csv, figures/*.png, tables/*.csv
└── tests/                         # pytest, mỗi module một file test_*.py
├── scripts/                           # scripts tiện ích tự động hóa
└── reports/                           # báo cáo và log nghiệm thu
└── README.md                      # tóm tắt dự án, cách chạy, thứ tự đọc
└── .gitignore
```

---

## 2. Schema dùng chung và hợp đồng kỹ thuật (khóa 100%)

### 2.1 File dữ liệu lõi

Đường dẫn thật: `src/data/`. Đường dẫn mock: `02_SAMPLE_DATA/mock_small/` và `mock_fullshape/` (cùng tên file, cùng dtype, chỉ khác N).

| File | Shape | dtype | Nội dung | Ràng buộc kiểm tra |
|---|---|---|---|---|
| `P_val.npy` | `(2066, 28)` | `float64` | Xác suất sigmoid của ViSoBERT trên Val | `0.0 ≤ P ≤ 1.0`, không NaN/Inf, C-contiguous |
| `Y_val.npy` | `(2066, 28)` | `uint8` | Nhãn thật multi-hot trên Val | Giá trị chỉ `{0,1}`; mỗi hàng có ≥ 1 nhãn 1 |
| `P_test.npy` | `(2066, 28)` | `float64` | Xác suất sigmoid trên Test | Như `P_val.npy` |
| `Y_test.npy` | `(2066, 28)` | `uint8` | Nhãn thật trên Test | Như `Y_val.npy` |
| `meta.json` | — | JSON | Xem 2.2 | Có sha256 mỗi file |

Quy ước: **cột `c` của mọi ma trận ứng với `label_names[c]`**, hàng `i` của `P_*` và `Y_*` cùng một bình luận. Tên biến trong code luôn viết hoa `P`, `Y`.

### 2.2 `meta.json`

```json
{
  "contract_version": "1.0",
  "dataset": "ViGoEmotions",
  "n_total": 20664, "n_val": 2066, "n_test": 2066, "n_labels": 28,
  "label_names": ["...28 tên theo đúng thứ tự cột..."],
  "neutral_idx": 27,
  "model": {"name": "ViSoBERT", "lr": 5e-5, "dropout": 0.2, "epochs": 12,
            "loss": "BCEWithLogits+pos_weight", "ckpt_selection": "last_epoch|best_val (ghi rõ)",
            "git_commit": "<hash>", "train_seed": 42},
  "sha256": {"P_val.npy": "...", "Y_val.npy": "...", "P_test.npy": "...", "Y_test.npy": "..."},
  "baseline_at_0.5": {"val_macro_f1": 0.0, "test_macro_f1": 0.0,
                      "val_macro_precision": 0.0, "val_macro_recall": 0.0},
  "created_at": "ISO-8601", "created_by": "Thy"
}
```

`neutral_idx = 27` (cột cuối, khớp Bảng 4 của báo cáo: Neutral là nhãn thứ 28).  
Mốc tái lập baseline tại ngưỡng 0.5 (theo báo cáo): Val Macro-F1 ≈ **62.33%**, Test ≈ **61.50%**, Val Macro-Precision ≈ **59.08%**, Macro-Recall ≈ **64.40%**. Dung sai vận hành ±0.5 điểm phần trăm. Ngoài dung sai: không tự sửa, báo Ngọc để quyết định.

### 2.3 Ba file mapping (`src/configs/mappings/`)

Chỉ Thy sinh ra, Ngọc đóng băng bằng sha256. Chỉ số nhãn dùng 0-based theo cột.

| `mapping_mode` | D | File | Quy tắc |
|---|---|---|---|
| `"global"` | 1 | (không cần file) | `t_c = θ[0]` cho mọi c |
| `"freq3"` | 3 | `mapping_freq3.json` | Sắp xếp 28 nhãn giảm dần theo số mẫu dương ở **tập train**. Top 9 là Head (`0`), 10 nhãn kế là Mid (`1`), 9 nhãn cuối là Tail (`2`). Neutral xếp theo tần suất như mọi nhãn khác. Hòa tần suất: chỉ số cột nhỏ hơn xếp trước. |
| `"semantic4"` | 4 | `mapping_semantic4.json` | Nhóm `[pos, amb, neg, neu]` theo nhóm sắc thái kiểu GoEmotions. Nga đối chiếu từng nhãn với danh sách nhãn ViGoEmotions và ghi vào `mapping_semantic4.json` kèm nguồn. |
| `"per_label"` | 28 | (không cần file) | `t_c = θ[c]` |

Schema `mapping_freq3.json`: `{"group_of_label": [28 số 0/1/2], "head_idx": [...], "mid_idx": [...], "tail_idx": [...], "train_counts": [28 số], "source": "train split, seed ..."}`.  
`semantic4` dùng `group_of_label` với giá trị `0=pos, 1=amb, 2=neg, 3=neu`.  
Tra cứu D từ mode: `D_OF_MODE = {"global":1, "freq3":3, "semantic4":4, "per_label":28}` đặt trong `src/contract/constants.py`.

### 2.4 Hàm ra quyết định — chữ ký bị khóa

Vị trí: `src/core/decision.py`. Owner: Thy.

```python
import numpy as np

MAPPING_MODES = ("global", "freq3", "semantic4", "per_label")
FALLBACK_STRATEGIES = ("argmax", "neutral", "head_argmax", "none")

def expand_theta(theta: np.ndarray, mapping_mode: str) -> np.ndarray:
    """theta (D,) -> t (28,) theo mapping_mode. Raise ValueError nếu len(theta) != D_OF_MODE[mapping_mode]."""

def predict_labels(
    P: np.ndarray,                    # (N, 28) float64 trong [0, 1]
    theta: np.ndarray,                # (D,) float64 trong [0.05, 0.95]
    mapping_mode: str = "freq3",      # một trong MAPPING_MODES
    fallback_strategy: str = "argmax",# một trong FALLBACK_STRATEGIES
    exclusive_neutral: bool = True,
) -> np.ndarray:                      # (N, 28) uint8, giá trị {0, 1}
    ...
```

Ngữ nghĩa bắt buộc (theo thứ tự):

1. `t = expand_theta(theta, mapping_mode)`; `S = (P >= t)`. Dấu so sánh là **`>=`** (xác suất bằng ngưỡng thì dự đoán 1).
2. Nếu `exclusive_neutral` (LD-11): với mỗi hàng `i` có `S[i, neutral_idx] == 1` và `S[i].sum() > 1`:
   - `m_i = max{ P[i, c] : c ∈ S_i, c ≠ neutral_idx }`.
   - Nếu `P[i, neutral_idx] > m_i` (so sánh **chặt**): **giữ Neutral, loại tất cả cảm xúc khác** trong `S_i` (hàng chỉ còn `neutral_idx`).
   - Ngược lại (`P[i, neutral_idx] ≤ m_i`, kể cả bằng nhau): **loại Neutral**, giữ nguyên các cảm xúc còn lại.
   - Hàng chỉ có Neutral (`|S_i| = 1`) giữ nguyên. Bước này không bao giờ làm hàng trở nên rỗng.
   Gợi ý vector hóa (không vòng lặp theo hàng): `other = S.copy(); other[:, n] = False; m = np.where(other, P, -np.inf).max(1); multi = S[:, n] & other.any(1); win = multi & (P[:, n] > m); S[win] = False; S[win, n] = True; S[multi & ~win, n] = False`.
3. Với mỗi hàng rỗng (`S[i].sum() == 0`), áp dụng `fallback_strategy`:
   - `"argmax"`: bật `argmax_c P[i, c]` trên cả 28 nhãn.
   - `"neutral"`: bật `neutral_idx`.
   - `"head_argmax"`: bật `argmax` trong tập `head_idx` của `mapping_freq3.json` (luôn dùng Head theo tần suất, không phụ thuộc `mapping_mode`).
   - `"none"`: giữ hàng rỗng.
4. Hàm thuần túy (không side effect), vector hóa hoàn toàn (không vòng lặp Python theo hàng), không sửa `P`.

Kiểm tra đầu vào bắt buộc: shape, dtype, dải giá trị. Lỗi ném `ValueError` với thông điệp nêu rõ tham số sai.

### 2.5 Hàm mục tiêu

Vị trí: `src/core/objective.py`. Owner: Thy.

```python
def per_label_f1(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    """(28,) F1_c = 2TP / (2TP + FP + FN); trả 0.0 khi mẫu số = 0 (tương đương zero_division=0)."""

def macro_f1(Y: np.ndarray, Yhat: np.ndarray) -> float: ...

def make_objective(P, Y, mapping_mode, fallback_strategy, exclusive_neutral,
                   lam: float, t_g: float) -> "Callable[[np.ndarray], float]":
    """Trả về f(theta) -> J_λ(theta) (float, càng lớn càng tốt).
       J_λ = macro_f1(Y, predict_labels(P, theta, ...)) - lam * mean((theta - t_g)**2)  # mean = (1/D)·‖·‖²
       Với mapping_mode == "global": ép lam = 0."""
```

Yêu cầu hiệu năng: `f(theta)` ≤ **3 ms** trung vị trên ma trận `(2066, 28)`, 1 luồng. Đếm TP/FP/FN bằng phép nhân/cộng ma trận, không dùng `sklearn` trong vòng lặp nóng.

### 2.6 Bộ đếm ngân sách `NFETracker` — mã tham chiếu bị khóa

Vị trí: `src/contract/nfe_tracker.py`. Ngọc commit nguyên văn, không ai được sửa nếu chưa có Change Request. Mã này đã được chạy thử: ngắt đúng tại `max_nfe`, ném `BudgetExhausted` **trước** khi đánh giá lần gọi thứ `max_nfe+1`, kết quả tái lập từng bit khi cùng seed.

```python
from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple

LO, HI = 0.05, 0.95
OFFICIAL_BUDGETS = (200, 1000, 5000)

class BudgetExhausted(Exception):
    """Ném TRƯỚC khi đánh giá lần gọi thứ max_nfe+1."""

class NFETracker:
    def __init__(self, objective: Callable[[np.ndarray], float], max_nfe: int,
                 surrogate: Optional[Callable[[np.ndarray, float], Tuple[float, np.ndarray]]] = None,
                 bounds=(LO, HI), enforce_official: bool = True):
        if enforce_official and max_nfe not in OFFICIAL_BUDGETS:
            raise ValueError(f"max_nfe must be in {OFFICIAL_BUDGETS}")
        self.objective, self.surrogate = objective, surrogate
        self.max_nfe, self.lo, self.hi = int(max_nfe), bounds[0], bounds[1]
        self.nfe = 0
        self.best_score, self.best_theta = -np.inf, None
        self._raw, self._best, self._kind = [], [], []

    @property
    def remaining(self) -> int:
        return self.max_nfe - self.nfe

    def _charge(self):
        if self.nfe >= self.max_nfe:
            raise BudgetExhausted()
        self.nfe += 1

    def _clip(self, theta):
        return np.clip(np.asarray(theta, dtype=np.float64).ravel(), self.lo, self.hi)

    def __call__(self, theta) -> float:               # đánh giá J_λ rời rạc, 1 NFE
        self._charge()
        th = self._clip(theta)                        # tracker luôn clip vào [0.05, 0.95]
        s = float(self.objective(th))
        if s > self.best_score:                       # strict >: giữ nghiệm gặp đầu tiên
            self.best_score, self.best_theta = s, th.copy()
        self._raw.append(s); self._best.append(self.best_score); self._kind.append(0)
        return s

    def evaluate_surrogate(self, theta, tau: float):  # surrogate + gradient, 1 NFE
        if self.surrogate is None:
            raise RuntimeError("tracker was built without surrogate")
        self._charge()
        th = self._clip(theta)
        val, grad = self.surrogate(th, tau)
        self._raw.append(np.nan)
        self._best.append(self.best_score if self.best_theta is not None else np.nan)
        self._kind.append(1)
        return float(val), np.asarray(grad, dtype=np.float64)

    def trajectory(self) -> dict:
        return {"raw_scores": np.asarray(self._raw, dtype=np.float64),      # NaN tại lần gọi surrogate
                "best_so_far": np.asarray(self._best, dtype=np.float64),    # độ dài = actual_nfe
                "call_kind": np.asarray(self._kind, dtype=np.uint8)}        # 0 = rời rạc, 1 = surrogate
```

Quy tắc đếm:

- `best_score`/`best_theta` **chỉ** cập nhật từ đánh giá rời rạc `J_λ`. Điểm số ghi vào CSV luôn là `J_λ` rời rạc, kể cả với Adam/L-BFGS-B.
- Optimizer không được giữ bản sao hàm mục tiêu gốc để gọi vòng qua tracker. Ngọc grep `make_objective` trong `src/optimizers/` khi review; chỉ được xuất hiện trong `src/runner/`.
- Thời gian chạy của sampler (ví dụ TPE) **không** tính vào NFE nhưng **có** tính vào `time_seconds`.

### 2.7 Lớp cơ sở `BaseOptimizer` — mã tham chiếu bị khóa

Vị trí: `src/contract/base_optimizer.py`. Mọi thuật toán kế thừa lớp này, chỉ cài đặt `_search`; `optimize()` là template method, **không override**.

```python
from __future__ import annotations
import time, zlib
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from .nfe_tracker import NFETracker, BudgetExhausted, LO, HI

def init_theta(seed: int, D: int, bounds=(LO, HI)) -> np.ndarray:
    """Điểm khởi tạo dùng chung (LD-5)."""
    return np.random.default_rng([seed, D, 0]).uniform(bounds[0], bounds[1], size=D)

class BaseOptimizer(ABC):
    name: str = "base"                       # snake_case, phải khớp registry (mục 2.8)

    def __init__(self, D: int, bounds=(LO, HI), **hparams):
        self.D, self.bounds, self.hparams = int(D), bounds, hparams

    @abstractmethod
    def _search(self, tracker: NFETracker, x0: np.ndarray, rng: np.random.Generator) -> bool:
        """Chạy tìm kiếm, mọi đánh giá đi qua `tracker`. Trả True nếu DỪNG vì tiêu chí hội tụ
        nội bộ trước khi hết budget, False nếu chạy đến khi hết budget."""

    def optimize(self, tracker: NFETracker, seed: int) -> Tuple[np.ndarray, float, dict]:
        rng = np.random.default_rng([seed, self.D, zlib.crc32(self.name.encode()), 1])
        x0 = init_theta(seed, self.D, self.bounds)
        t0 = time.perf_counter()
        try:
            conv = bool(self._search(tracker, x0, rng))
        except BudgetExhausted:
            conv = False
        elapsed = time.perf_counter() - t0
        if tracker.best_theta is None:
            raise RuntimeError("optimizer made no discrete evaluation")
        traj = tracker.trajectory()
        bsf = traj["best_so_far"]
        tol = 0.01 * abs(tracker.best_score)
        hit = np.where(~np.isnan(bsf) & (bsf >= tracker.best_score - tol))[0]
        traj.update(
            actual_nfe=tracker.nfe, nfe_budget=tracker.max_nfe,
            is_converged=bool(conv and tracker.remaining > 0),
            time_seconds=elapsed,
            nfe_to_99pct=int(hit[0]) + 1 if hit.size else None,   # NFE đầu tiên đạt 99% giá trị cuối
            extra=getattr(self, "extra", {}),                      # ví dụ diversity theo thế hệ
        )
        return tracker.best_theta.copy(), float(tracker.best_score), traj
```

**Đầu ra `optimize()` bị khóa:** `(best_theta, best_score, trajectory_dict)`.

| Phần tử | Kiểu | Ràng buộc |
|---|---|---|
| `best_theta` | `np.ndarray (D,) float64` | Nằm trong `[0.05, 0.95]` |
| `best_score` | `float` | `J_λ` rời rạc tại `best_theta` |
| `trajectory_dict` | `dict` | Khóa bắt buộc: `raw_scores`, `best_so_far`, `call_kind`, `actual_nfe`, `nfe_budget`, `is_converged`, `time_seconds`, `nfe_to_99pct`, `extra` |

`extra` là nơi duy nhất để thuật toán đính kèm dữ liệu riêng (ví dụ `extra["diversity"]` cho DE/PSO, `extra["tau_schedule"]` cho gradient). Không thêm khóa mới ở cấp ngoài.

### 2.8 Registry thuật toán và dải chạy

`algorithm_name` chỉ lấy từ danh sách sau (đặt trong `src/runner/registry.py`):

| `algorithm_name` | Nhóm DS106 | Owner | D chạy | Ghi chú |
|---|---|---|---|---|
| `constant_0.5` | Baseline tĩnh | Nga (runner), Thy (hàm) | 1 | Một dòng duy nhất: `seed=0`, `nfe_budget=1`, `actual_nfe=1` |
| `random_search` | Baseline NFLT (Ch.4) | Nga | 1, 3, 28 | |
| `coordinate_ascent` | Local (Ch.3, 5) | Thy | 1, 3, 28 | |
| `nelder_mead` | Local (Ch.3, 5) | Thy | 1, 3, 28 | |
| `powell` | Local (Ch.3, 5) | Thy | 1, 3, 28 | |
| `simulated_annealing` | Stochastic (Ch.3, 5) | Nga | 1, 3, 28 | |
| `differential_evolution` | Quần thể (Ch.3, 5, 6) | Nga | 1, 3, 28 | |
| `pso` | Quần thể (Ch.3, 5, 6) | Nga | 1, 3, 28 | |
| `optuna_tpe` | Bayes (Ch.2, 3) | Nga | 1, 3, 28 | |
| `adam_surrogate` | Gradient bậc 1 (Ch.3) | Vinh | 1, 3, 28 | |
| `lbfgsb_surrogate` | Gradient bậc 1 (Ch.3) | Vinh | 1, 3, 28 | |

Số dòng kỳ vọng của `benchmark_results.csv`: `10 thuật toán × 3 D × 20 seeds × 3 budgets + 1 = 1801`.

Ánh xạ D → `mapping_mode` trong benchmark chính: `1→"global"`, `3→"freq3"`, `28→"per_label"`. Biến thể `"semantic4"` chỉ chạy ở ablation (EVAL-01).

### 2.9 Schema log thực nghiệm `benchmark_results_schema.csv`

File thật: `src/outputs/benchmark_results.csv`. Header **đúng thứ tự, đúng chính tả**, không thêm cột, không đổi tên:

```csv
algorithm_name,dimension_D,seed,nfe_budget,actual_nfe,best_val_score,time_seconds,is_converged
```

| Cột | Kiểu | Quy tắc |
|---|---|---|
| `algorithm_name` | string | Thuộc registry 2.8 |
| `dimension_D` | int | `{1, 3, 28}` |
| `seed` | int | `0..19` |
| `nfe_budget` | int | `{200, 1000, 5000}`; riêng `constant_0.5` là `1` |
| `actual_nfe` | int | `1 ≤ actual_nfe ≤ nfe_budget`. Nếu `is_converged=False` thì bắt buộc `actual_nfe == nfe_budget` |
| `best_val_score` | float | `J_λ*` rời rạc tại `best_theta`, in 10 chữ số thập phân |
| `time_seconds` | float | Wall-clock `perf_counter` của `optimize()` (6 chữ số thập phân) |
| `is_converged` | bool | Ghi `True`/`False` |

Khóa chính: `(algorithm_name, dimension_D, seed, nfe_budget)`; **không được trùng**.

Tệp đi kèm bắt buộc (sidecar), vì CSV không chứa vector ngưỡng:

- `src/outputs/benchmark_theta_log.jsonl`: mỗi dòng `{"algorithm_name","dimension_D","seed","nfe_budget","best_theta":[...],"lambda":x,"nfe_to_99pct":n}`.
- `src/outputs/trajectories/{algorithm_name}_D{D}_s{seed}_n{budget}.npz`: chứa `best_so_far`, `call_kind`, và `extra`.

### 2.10 Seeds, tái lập và môi trường

- Python 3.11. Pin toàn bộ phiên bản bằng `pip freeze` ở M0 (numpy, scipy, optuna, pandas, matplotlib, scikit-learn, iterative-stratification, pytest).
- Đặt `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` trước khi import numpy. Song song hóa chỉ ở mức tiến trình (runner), không ở mức BLAS.
- Cấm `np.random.seed` toàn cục và cấm Python `hash()`. Dùng `np.random.default_rng([...])` và `zlib.crc32` như trong 2.7.
- Chạy lại cùng khóa `(algorithm, D, seed, budget)` phải cho `best_val_score` **giống từng bit** (TPE: `n_jobs=1`).

### 2.11 Quy tắc sai số kiểm thử

- So sánh số thực trong test: `np.isclose(a, b, rtol=0, atol=1e-12)` cho Macro-F1; `atol=1e-9` khi đối chiếu với `sklearn`.
- Mọi so sánh dự đoán nhãn là **bằng nhau tuyệt đối** (`np.array_equal`).

### 2.12 Cô lập tập Test (chốt chặn của Ngọc)

1. Chỉ `src/eval/test_gate.py` được phép `np.load` các file `*_test*.npy`. Không module nào khác chứa chuỗi `P_test`, `Y_test`, `_test.npy`.
2. `test_gate.load_test()` từ chối chạy trừ khi tồn tại `src/artifacts/LOCK_FREEZE.json` và sha256 của `benchmark_results.csv` cùng `benchmark_theta_log.jsonl` khớp với giá trị trong file đó. Chỉ Ngọc tạo `LOCK_FREEZE.json` sau khi nghiệm thu benchmark (mục LEAD-01 G4).
3. Cờ `--use_mock` chuyển cổng sang `02_SAMPLE_DATA/mock_*/` và bỏ qua kiểm tra freeze. Cờ này bị cấm trong đường chạy dùng cho báo cáo.
4. CI kiểm tra: `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ được trả về `src/eval/test_gate.py` và file test của chính nó.
5. Tuning λ, tuning siêu tham số và chọn `θ*` chỉ dùng Val.

### 2.13 Change Request

Sửa bất kỳ mục nào trong 2.1–2.12 phải có mục trong `CHANGE_REQUESTS.md` (ai, lý do, tác động lên file/module nào, phiên bản mới) và chữ ký của Ngọc. Chưa có chữ ký thì mã theo bản cũ.

---