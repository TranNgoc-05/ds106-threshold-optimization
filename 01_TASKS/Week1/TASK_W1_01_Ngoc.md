# TASK 01 — Ngọc — Tuần 1: Hợp đồng, khung mã nguồn và PoC

**Vai trò:** Technical Project Manager & System Architect  
**Owner:** Ngọc  
**Reviewers:** Thy (Core / Local), Nga (Global / Runner), Vinh (Surrogate / Eval)  
**Deadline tuần:** D+6 23:30  
**Mốc trong tuần:** M0 (hợp đồng) D+1 12:00 · M0b (mock data) D+1 23:30 · M1 (cache ViSoBERT, decision, objective) D+3 23:30 · Slide PoC D+5 23:30 · Báo cáo sơ bộ với Giảng viên D+6

> **Quy ước thời gian.** D0 là ngày đầu Tuần 1. Tuần n gồm các ngày từ `D+7(n−1)` đến `D+7n−1`. Lịch 12 ngày trong Task Package gốc được dàn lại theo 4 tuần của kế hoạch Word.
>
> **Đồng bộ phân vai.** Thy phụ trách Core/Local (ViSoBERT cache, `decision.py`, `objective.py`, landscape, `coordinate_ascent`, `nelder_mead`, `powell`). Nga phụ trách Global/Runner (`random_search`, `simulated_annealing`, `differential_evolution`, `pso`, `optuna_tpe`, `run_benchmark.py`, `validate_results.py`). Vinh phụ trách Surrogate/Eval. Task Package v1.0 ghi hai vai Core và Global theo thứ tự ngược lại; Ngọc ghi CR-004 ở mục 3.5 để sửa.

## 1. Mục tiêu tuần (Weekly Goal)

1. Ban hành hợp đồng kỹ thuật bất biến `CONTRACT_v1.0.md` cùng khung mã nguồn `src/contract/` để cả nhóm lập trình trên cùng một interface.
2. Dựng hàng rào tự động: kiểm thử hợp đồng, CI chặn rò rỉ tập Test, chặn gọi hàm mục tiêu vòng qua tracker.
3. Đăng ký trước tiêu chí kiểm chứng H1–H4 và No Free Lunch (NFLT) vào `PREREG.md` trước khi có bất kỳ kết quả benchmark nào.
4. Nghiệm thu Cổng G1 cho pipeline dữ liệu thật của Thy và bản mock runner của Nga.
5. Tổng hợp PoC trên tập Validation (địa hình hàm số, ngưỡng 0.5 so với Random Search, Coordinate Ascent và Optuna TPE) và báo cáo sơ bộ với Giảng viên.

## 2. Input đầu vào

| Nguồn | Tài liệu / sản phẩm | Dùng để |
|---|---|---|
| Nhóm | `DS106_Sprint1_Task_Package.md` (bản đã chốt K-1…K-5), file Word kế hoạch 4 tuần | Nguồn sự thật của hợp đồng và lịch |
| Nhóm | Báo cáo đề xuất đồ án DS106 | Đối chiếu công thức `J_λ` (mục 4.3) |
| Giảng viên | Slide DS106 (Nguyễn-Tất B. Thiện, 2026), Chapter 1–6 | Cơ sở lý thuyết cho PoC và báo cáo |
| Thy | `P_val.npy`, `Y_val.npy`, `P_test.npy`, `Y_test.npy`, `meta.json`, `mapping_freq3.json`, `mapping_semantic4.json`, `t_g.json`, `make_mock_data.py`, `decision.py`, `objective.py`, biểu đồ địa hình 1D/2D, kết quả Coordinate Ascent | Nghiệm thu G1, làm PoC |
| Nga | Mock runner, kết quả pilot 3–5 seeds của `random_search` và `optuna_tpe` | Nghiệm thu G1, làm PoC |
| Vinh | Bản nháp `surrogate.py` trên mock (chưa bắt buộc nộp tuần này) | Biết sớm các ràng buộc về interface |

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Khởi tạo kho mã nguồn (M0, D+1 sáng)

```bash
git init ds106-threshold-opt && cd ds106-threshold-opt
mkdir -p 00_START_HERE 01_TASKS 02_SAMPLE_DATA 03_SUBMISSIONS \
         src/{contract,core,optimizers/{local,global_,gradient},runner,eval,configs/mappings,artifacts,data,outputs,training} \
         tests scripts reports .github/workflows
touch src/__init__.py src/contract/__init__.py scripts/__init__.py
printf 'src/data/\n*.pt\n*.ckpt\n__pycache__/\n.env\nsrc/outputs/trajectories/\n' > .gitignore
git add -A && git commit -m "chore: cấu trúc thư mục chuẩn contract v1.0"
```

Bật bảo vệ nhánh `main`: bắt buộc PR, bắt buộc CI xanh, không push trực tiếp, bắt buộc một review của Ngọc (Cổng G1).

### 3.2 Commit file lõi trong `src/contract/` (M0)

1. `nfe_tracker.py`: sao chép **nguyên văn** mã ở mục 2.6 của hợp đồng (lớp `NFETracker`, ngoại lệ `BudgetExhausted`, hằng `LO`, `HI`, `OFFICIAL_BUDGETS`). Không chỉnh sửa dù chỉ một ký tự; mã đã được kiểm thử ngắt đúng tại `max_nfe`.
2. `base_optimizer.py`: sao chép **nguyên văn** mã ở mục 2.7 (hàm `init_theta`, lớp trừu tượng `BaseOptimizer` với template method `optimize()` trả về `(best_theta, best_score, trajectory_dict)`).
3. `constants.py`: nhập lại `LO`, `HI`, `OFFICIAL_BUDGETS` từ `nfe_tracker.py` để chỉ có một nguồn duy nhất, không lặp định nghĩa:

```python
from .nfe_tracker import LO, HI, OFFICIAL_BUDGETS

N_LABELS = 28
NEUTRAL_IDX = 27
MAPPING_MODES = ("global", "freq3", "semantic4", "per_label")
D_OF_MODE = {"global": 1, "freq3": 3, "semantic4": 4, "per_label": 28}
FALLBACK_STRATEGIES = ("argmax", "neutral", "head_argmax", "none")
BENCHMARK_DIMS = (1, 3, 28)
BENCHMARK_SEEDS = tuple(range(20))
TUNING_SEEDS = tuple(range(100, 105))
LANDSCAPE_SEEDS = tuple(range(200, 400))
ALGO_REGISTRY_NAMES = (
    "constant_0.5", "random_search", "coordinate_ascent", "nelder_mead", "powell",
    "simulated_annealing", "differential_evolution", "pso", "optuna_tpe",
    "adam_surrogate", "lbfgsb_surrogate",
)
CSV_COLUMNS = ("algorithm_name", "dimension_D", "seed", "nfe_budget",
               "actual_nfe", "best_val_score", "time_seconds", "is_converged")
EXPECTED_ROWS_FULL = 10 * len(BENCHMARK_DIMS) * len(BENCHMARK_SEEDS) * len(OFFICIAL_BUDGETS) + 1  # = 1801
```

### 3.3 Bộ kiểm thử hợp đồng `tests/test_contract.py` (M0)

Bộ test này đã được chạy thử và đạt 13/13. Các hành vi bắt buộc được kiểm tra: ngắt đúng tại budget, từ chối budget ngoài danh sách, clip biên, quy tắc `best` chặt, surrogate tốn 1 NFE và không cập nhật `best`, hội tụ sớm, tái lập từng bit, độ dài quỹ đạo, bộ ba trả về, tính nhất quán của hằng số.

```python
import numpy as np
import pytest

from src.contract.base_optimizer import BaseOptimizer, init_theta
from src.contract.constants import (ALGO_REGISTRY_NAMES, D_OF_MODE, EXPECTED_ROWS_FULL, HI, LO,
                                    MAPPING_MODES, OFFICIAL_BUDGETS)
from src.contract.nfe_tracker import BudgetExhausted, NFETracker


class CountingObjective:
    """Bộ đếm độc lập, nằm ngoài NFETracker."""

    def __init__(self):
        self.calls = 0

    def __call__(self, theta):
        self.calls += 1
        return float(-np.sum((theta - 0.6) ** 2))


class RandomSearchStub(BaseOptimizer):
    name = "random_search"

    def _search(self, tracker, x0, rng):
        while True:
            tracker(rng.uniform(LO, HI, size=self.D))


class EarlyStopStub(BaseOptimizer):
    name = "early_stub"

    def _search(self, tracker, x0, rng):
        for _ in range(7):
            tracker(x0)
        return True


@pytest.mark.parametrize("budget", OFFICIAL_BUDGETS)
def test_tracker_stops_exactly_at_budget(budget):
    obj = CountingObjective()
    tracker = NFETracker(obj, budget)
    for _ in range(budget):
        tracker(np.full(3, 0.5))
    with pytest.raises(BudgetExhausted):
        tracker(np.full(3, 0.5))
    assert obj.calls == budget
    assert tracker.nfe == budget and tracker.remaining == 0


def test_tracker_rejects_unofficial_budget():
    with pytest.raises(ValueError):
        NFETracker(CountingObjective(), 300)
    NFETracker(CountingObjective(), 300, enforce_official=False)


def test_tracker_clips_into_bounds_and_strict_best():
    seen = []
    tracker = NFETracker(lambda th: (seen.append(th.copy()) or 1.0), 200)
    tracker(np.array([-5.0, 9.0]))
    assert np.all(seen[0] >= LO) and np.all(seen[0] <= HI)
    first_best = tracker.best_theta.copy()
    tracker(np.array([0.5, 0.5]))  # cùng điểm số 1.0, không được thay best (strict >)
    assert np.array_equal(tracker.best_theta, first_best)


def test_surrogate_call_costs_one_nfe_and_does_not_set_best():
    tracker = NFETracker(lambda th: 0.5, 200, surrogate=lambda th, tau: (0.9, np.zeros_like(th)))
    tracker(np.full(3, 0.5))
    tracker.evaluate_surrogate(np.full(3, 0.5), 0.1)
    traj = tracker.trajectory()
    assert tracker.nfe == 2 and tracker.best_score == 0.5
    assert list(traj["call_kind"]) == [0, 1]


def test_early_stop_reports_converged():
    tracker = NFETracker(CountingObjective(), 200)
    _, _, traj = EarlyStopStub(D=3).optimize(tracker, seed=0)
    assert traj["is_converged"] is True
    assert traj["actual_nfe"] == 7 < traj["nfe_budget"] == 200


def test_budget_exhausted_is_not_converged():
    tracker = NFETracker(CountingObjective(), 200)
    _, _, traj = RandomSearchStub(D=3).optimize(tracker, seed=0)
    assert traj["is_converged"] is False and traj["actual_nfe"] == 200


def test_bit_exact_same_seed():
    runs = []
    for _ in range(2):
        tracker = NFETracker(CountingObjective(), 1000)
        theta, score, _ = RandomSearchStub(D=28).optimize(tracker, seed=7)
        runs.append((theta.tobytes(), score))
    assert runs[0] == runs[1]


def test_trajectory_length_matches_actual_nfe():
    obj = CountingObjective()
    tracker = NFETracker(obj, 200)
    _, _, traj = RandomSearchStub(D=3).optimize(tracker, seed=1)
    assert len(traj["best_so_far"]) == len(traj["raw_scores"]) == traj["actual_nfe"] == obj.calls
    for key in ("raw_scores", "best_so_far", "call_kind", "actual_nfe", "nfe_budget",
                "is_converged", "time_seconds", "nfe_to_99pct", "extra"):
        assert key in traj


def test_init_theta_deterministic_and_in_bounds():
    a, b = init_theta(3, 28), init_theta(3, 28)
    assert np.array_equal(a, b) and a.shape == (28,) and np.all((a >= LO) & (a <= HI))
    assert not np.array_equal(init_theta(3, 28), init_theta(4, 28))


def test_optimize_returns_locked_tuple():
    out = RandomSearchStub(D=3).optimize(NFETracker(CountingObjective(), 200), seed=0)
    assert len(out) == 3 and isinstance(out[0], np.ndarray) and isinstance(out[1], float) and isinstance(out[2], dict)


def test_constants_are_consistent():
    assert (LO, HI) == (0.05, 0.95) and OFFICIAL_BUDGETS == (200, 1000, 5000)
    assert set(D_OF_MODE) == set(MAPPING_MODES)
    assert len(ALGO_REGISTRY_NAMES) == 11 and len(set(ALGO_REGISTRY_NAMES)) == 11
    assert EXPECTED_ROWS_FULL == 1801
```

### 3.4 CI và hàng rào chặn rò rỉ Test (M0)

**`scripts/check_test_isolation.sh`** (đã chạy thử trường hợp đạt và trường hợp vi phạm). Script chặn ba lỗi: chuỗi liên quan tập Test ngoài danh sách cho phép, `make_objective` trong `src/optimizers/`, và `np.random.seed` hoặc `hash()`:

```bash
#!/usr/bin/env bash
# Cổng cách ly Test: chỉ các file trong ALLOWED được chứa chuỗi liên quan tập Test.
set -uo pipefail
ALLOWED="src/eval/test_gate.py tests/test_test_gate.py"
HITS=$(grep -RIl "P_test\|Y_test\|_test\.npy" src tests --include=*.py 2>/dev/null | sort)
STATUS=0
for f in $HITS; do
  case " $ALLOWED " in
    *" $f "*) ;;
    *) echo "VI PHAM cach ly Test: $f"; STATUS=1 ;;
  esac
done
if grep -RIn "make_objective" src/optimizers --include=*.py 2>/dev/null; then
  echo "VI PHAM: make_objective xuat hien trong src/optimizers"; STATUS=1
fi
if grep -RInE "np\.random\.seed|[^_a-zA-Z.]hash\(" src --include=*.py 2>/dev/null; then
  echo "VI PHAM: dung np.random.seed hoac hash()"; STATUS=1
fi
[ $STATUS -eq 0 ] && echo "Cach ly Test: OK"
exit $STATUS
```

**Hệ quả cho nhóm (ghi vào `SUBMISSION_RULES.md`):** script huấn luyện và nạp dữ liệu không được chứa chuỗi `P_test`, `Y_test`, `_test.npy`. Khi xuất hoặc nạp ma trận phải dùng tên sinh từ biến, ví dụ `np.save(out / f"P_{split}.npy", ...)`. Chỉ `src/eval/test_gate.py` được nạp dữ liệu Test.

**`.github/workflows/ci.yml`:**

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      OMP_NUM_THREADS: "1"
      MKL_NUM_THREADS: "1"
      OPENBLAS_NUM_THREADS: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r 00_START_HERE/requirements.txt
      - run: pytest -q
      - run: bash scripts/check_test_isolation.sh
```

### 3.5 `CHANGE_REQUESTS.md` và các CR bắt buộc

| Mã | Nội dung | Tác động | Trạng thái |
|---|---|---|---|
| CR-001 | OI-1 chốt phương án (a): giữ `J_λ` theo LD-8. Phạt L2 là regularization kéo nghiệm về `t_g`; coupling đến hoàn toàn từ Neutral độc quyền và Fallback | Báo cáo, README, `ablation` | Ngọc ký |
| CR-002 | Bỏ hoàn toàn CafeBERT; chỉ dùng ViSoBERT | Schema `02_SAMPLE_DATA`, `EVAL-01`, Bảng 2, rủi ro | Ngọc ký |
| CR-003 | Quy tắc Neutral độc quyền theo so sánh xác suất chặt | `decision.py`, golden cases, differential test, ablation | Ngọc ký |
| CR-004 | Đồng bộ phân vai: Thy = Core/Local, Nga = Global/Runner, Vinh = Surrogate/Eval | Đổi owner `CORE-01` và `GLOBAL-01` trong `01_TASKS/`, ma trận cross-review | Ngọc ký sau khi nhóm xác nhận |
| CR-005 | Bổ sung định dạng tối thiểu cho log: `tuning_log.csv` thêm cột `nfe_budget`, `split`; `cv_lambda_results.csv` có cột `dimension_D, lam, mean_heldout_macro_f1, se`; `registry.py` xuất biến `REGISTRY: Dict[str, Type[BaseOptimizer]]`; `selective_pressure_study.csv` có cột `algorithm, variant, seed, best_val_score, diversity_collapse_nfe`; `stats_mwu_holm.csv` dùng giá trị `primary` / `exploratory` cho cột `family` | Cho phép Ngọc kiểm toán G2, G3, G5 bằng script | Ngọc ký |

Câu chữ bắt buộc cho `README.md` và khung báo cáo:

- **Quy tắc Neutral độc quyền.** Khi `neutral ∈ S_i` và `|S_i| > 1`: nếu `P[i, neutral] > max P[i, c]` với mọi `c ∈ S_i \ {neutral}` thì giữ Neutral và loại mọi cảm xúc khác; ngược lại loại Neutral và giữ các cảm xúc còn lại (bằng nhau thì loại Neutral).
- **Phạt L2.** Số hạng phạt L2 đóng vai trò chống quá khớp (regularization), kéo nghiệm về `t_g`. Sự ghép cặp (coupling) giữa các nhãn đến hoàn toàn từ cơ chế Neutral độc quyền và Fallback.
- **Mô hình.** Dự án chỉ dùng một mô hình duy nhất là ViSoBERT.

### 3.6 Đăng ký trước `00_START_HERE/PREREG.md` (hạn D+4, trước M2)

Ngọc commit file dưới đây. **Sau commit đầu tiên không sửa nội dung**; nếu cần làm rõ thì thêm mục "Phụ lục" có dấu thời gian. Các điều khoản có chữ "đăng ký" là lựa chọn vận hành của Ngọc, có thể chỉnh trước khi commit.

```markdown
# PREREG — Đăng ký trước tiêu chí kiểm chứng (contract v1.0)

Commit của file này phải có dấu thời gian sớm hơn commit đầu tiên của bất kỳ file kết quả benchmark nào.

## Quy ước chung
- Mô hình duy nhất: ViSoBERT. Độ đo chính: Macro-F1. Mức ý nghĩa: alpha = 0.05.
- Kiểm định: Mann-Whitney U hai phía (`method="asymptotic"`, `use_continuity=True`), hiệu chỉnh Holm trong từng ô (D, nfe_budget).
- Họ chính (primary): mỗi thuật toán so với `random_search` trong cùng ô, m = 9. Họ khám phá (exploratory): mọi cặp trong ô, m = 45.
- Generalization Gap = Macro-F1 (Val) − Macro-F1 (Test), cả hai KHÔNG có số hạng phạt, tính trên cùng nghiệm.
- "Dẫn đầu một ô" = trung vị `best_val_score` cao nhất trong ô. "Không kém có ý nghĩa" = Holm-corrected p >= 0.05.
- Kết quả trái dự đoán vẫn được báo cáo nguyên văn.

## H1 — Độ lệch ngưỡng do pos_weight
Ủng hộ nếu đồng thời: (a) t_g > 0.5; (b) MacroF1_val(t_g) − MacroF1_val(0.5) > 0; (c) tại 0.5, Macro-Precision < Macro-Recall trên Val.

## H2 — Quá khớp không gian tìm kiếm và vai trò chính quy hóa
Ủng hộ nếu, theo trung vị trên 20 seeds ở NFE = 5000, Gap của cấu hình (D=28, λ=0) lớn hơn Gap của (D=3) và lớn hơn Gap của (D=28, λ=λ*).

## H3 — Coordinate Ascent và tính gần tách biến
Ủng hộ nếu đồng thời: (a) ở D=28, NFE=5000, `coordinate_ascent` không kém có ý nghĩa so với `differential_evolution` và so với `pso`; (b) `differential_evolution`, `pso` hoặc `optuna_tpe` dẫn đầu ít nhất một ô trong số {D=28 NFE=200, D=3 NFE=200, D=3 NFE=1000, D=3 NFE=5000}.

## H4 — Độ lệch của phương pháp gradient trên surrogate
Ủng hộ nếu, tại NFE=5000 và cả hai D ∈ {3, 28}: (a) trung vị `nfe_to_99pct` của `adam_surrogate` hoặc `lbfgsb_surrogate` nhỏ hơn trung vị của mọi thuật toán hộp đen; (b) trung vị `best_val_score` của thuật toán gradient đó thấp hơn thuật toán hộp đen tốt nhất với Holm-corrected p < 0.05. (Phạm vi "NFE=5000, D ∈ {3, 28}" là lựa chọn vận hành đăng ký trước.)

## NFLT — No Free Lunch
Ủng hộ nếu đồng thời: (a) có ít nhất 2 thuật toán khác nhau giữ hạng nhất trong 9 ô (D, nfe_budget); (b) `random_search` không đứng cuối ở ít nhất một ô.

## Phân tích khám phá (không dùng để kết luận H1–H4)
Phân tích hội tụ sớm (Chapter 6), chi phí thời gian thật của TPE, so sánh `freq3` với `semantic4`.
```

### 3.7 Hoàn thiện `00_START_HERE/`

| File | Nội dung bắt buộc |
|---|---|
| `README.md` | Mục tiêu, cấu trúc repo, thứ tự đọc, lệnh tái lập, ba câu chữ bắt buộc ở 3.5 |
| `CONTRACT_v1.0.md` | Bản sao mục 2 của Task Package; ghi chữ ký và ngày ban hành |
| `SUBMISSION_RULES.md` | Checklist PR (mục 5.1 Task Package), quy tắc đặt tên tham chiếu từ biến (`f"P_{split}.npy"`), quy định PR chưa tick đủ checklist bị trả lại |
| `CHANGE_REQUESTS.md` | Bảng ở 3.5 |
| `PREREG.md` | Nội dung ở 3.6 |
| `requirements.txt` | Chạy `pip freeze` sau khi cài: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `optuna`, `iterative-stratification`, `statsmodels`, `pyyaml`, `tabulate`, `pytest` |
| `env.example` | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` |

### 3.8 Theo dõi đường găng và nghiệm thu G1 trên dữ liệu thật (M0b → M1)

**Nghiệm thu `make_mock_data.py` (M0b, D+1).** Chạy hai lần, so sha256:

```bash
python 02_SAMPLE_DATA/make_mock_data.py && sha256sum 02_SAMPLE_DATA/mock_small/*.npy > /tmp/h1.txt
python 02_SAMPLE_DATA/make_mock_data.py && sha256sum 02_SAMPLE_DATA/mock_small/*.npy > /tmp/h2.txt
diff /tmp/h1.txt /tmp/h2.txt && echo "Mock tái lập được"
```

Cấu trúc mock phải khớp 2.1: `float64` cho `P`, `uint8` cho `Y`, đúng shape, không có file CafeBERT.

**Nghiệm thu ma trận xác suất thật (M1, D+3).** Script độc lập của Ngọc chỉ băm byte và đọc header của file Test, không đọc giá trị:

```python
import hashlib, json
import numpy as np
from sklearn.metrics import f1_score

meta = json.load(open("src/data/meta.json"))
splits = ("val", "test")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

for kind in ("P", "Y"):
    for split in splits:
        name = f"{kind}_{split}.npy"                        # tên sinh từ biến, không vi phạm grep
        assert sha256_file(f"src/data/{name}") == meta["sha256"][name], f"Sai sha256: {name}"
        header = np.load(f"src/data/{name}", mmap_mode="r") # chỉ header, không đọc giá trị
        assert header.shape == (2066, 28), (name, header.shape)
        assert header.dtype == (np.float64 if kind == "P" else np.uint8), (name, header.dtype)

# Chỉ tập Validation được đọc giá trị
P, Y = np.load("src/data/P_val.npy"), np.load("src/data/Y_val.npy")
assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1 and set(np.unique(Y)) <= {0, 1}
assert (Y.sum(1) >= 1).all()
f1_val = f1_score(Y, (P >= 0.5).astype(np.uint8), average="macro", zero_division=0)
assert abs(f1_val - meta["baseline_at_0.5"]["val_macro_f1"]) < 1e-9
assert abs(f1_val - 0.6233) <= 0.005, f"Baseline Val lệch mốc 62.33%: {f1_val:.4f}"
print("Dữ liệu ViSoBERT: HỢP LỆ; Macro-F1 Val @0.5 =", round(f1_val, 4))
```

Giá trị Test @0.5 chỉ đọc từ `meta.json` (do Thy tính); mốc kỳ vọng ≈ 61.50% với dung sai ±0.5 điểm phần trăm. Nếu vượt dung sai thì dừng tuần này và yêu cầu Thy giải trình trước khi chạy tiếp.

**Review PR của Thy (Core Decision và Objective) theo Cổng G1:**

1. Chữ ký `predict_labels(P, theta, mapping_mode, fallback_strategy, exclusive_neutral)` khớp 2.4. So sánh dùng `>=`; hàm thuần túy, vector hóa, không sửa `P`.
2. Quy tắc Neutral đúng CR-003: có test cho Neutral thắng (loại mọi cảm xúc), Neutral thua (loại Neutral), hòa (loại Neutral), Neutral đứng một mình, `exclusive_neutral=False`.
3. `golden_cases.json` có tối thiểu 15 ca; differential test với `predict_labels_slow` (1000 cặp × mọi tổ hợp) bằng nhau tuyệt đối.
4. `make_objective`: trung vị ≤ 3 ms trên `mock_fullshape` (xem log đo); `macro_f1` khớp `sklearn` ở `1e-9`; với `mapping_mode="global"` thì `lam` bị ép về 0.
5. `mapping_freq3.json` đủ 9/10/9 theo tần suất train, `mapping_semantic4.json` ghi nguồn đối chiếu nhãn. Ngọc ghi sha256 hai file vào `CHANGE_REQUESTS.md` để đóng băng.
6. `t_g.json` có đường cong 901 điểm; tie-break theo LD-9.

**Review PR của Nga (mock runner) theo Cổng G1:** CLI đúng danh sách cờ trong GLOBAL-01; `--use-mock` hoạt động; `--resume` không sinh dòng trùng; `--tag pilot` ghi vào file `*_pilot.csv`; CSV đúng 8 cột đúng thứ tự.

### 3.9 PoC và báo cáo sơ bộ với Giảng viên (D+5 → D+6)

**Phạm vi bắt buộc:** chỉ dùng tập Validation. Không báo cáo kết quả tối ưu trên Test. Giá trị Test @0.5 chỉ dùng như mốc tái lập baseline.

**Lệnh chạy pilot** (do Nga chạy, `λ = 0.1` cho D=3 và D=28, đánh dấu pilot):

```bash
python -m src.runner.run_benchmark --algos random_search coordinate_ascent optuna_tpe \
  --dims 3 28 --budgets 1000 --seeds 0-4 --lam 0.1 --tag pilot --n-jobs 4 --resume
```

**Bảng so sánh nhỏ trên Validation** (mean ± std qua 5 seeds, NFE=1000):

| Cấu hình | D=3 (freq3) | D=28 (per_label) |
|---|---|---|
| Ngưỡng 0.5 (`constant_0.5`) | giá trị | giá trị |
| `random_search` | mean ± std | mean ± std |
| `coordinate_ascent` | mean ± std | mean ± std |
| `optuna_tpe` | mean ± std | mean ± std |

Script tổng hợp (`scripts/poc_table.py`):

```python
import pandas as pd

df = pd.read_csv("outputs/benchmark_results_pilot.csv")
base = df[df["algorithm_name"] == "constant_0.5"]["best_val_score"].iloc[0]
body = df[(df["nfe_budget"] == 1000) & (df["algorithm_name"] != "constant_0.5")]
tab = (body.groupby(["algorithm_name", "dimension_D"])["best_val_score"]
           .agg(["mean", "std", "count"]).round(4).reset_index())
tab["gain_vs_0.5"] = (tab["mean"] - base).round(4)
print(f"Ngưỡng 0.5: {base:.4f}")
print(tab.to_markdown(index=False))
tab.to_csv("outputs/tables/poc_week1.csv", index=False)
```

**Bộ slide PoC (16:9, 8–10 slide):** (1) Bài toán và động lực; (2) Mô hình hóa: `Θ`, `J_λ`, quy tắc Neutral; (3) Kiến trúc hợp đồng và 5 cổng chất lượng; (4) Đường cong `J(t)` 1 chiều; (5) Lát cắt `(t_head, t_tail)` và plateau ratio; (6) Số cực trị địa phương từ 200 điểm khởi động; (7) Bảng so sánh PoC; (8) Rủi ro và kế hoạch Tuần 2–4. Mỗi con số trên slide phải trỏ được tới một dòng trong `landscape_summary.csv` hoặc `poc_week1.csv`.

## 4. Output tuần

| Nhóm | File |
|---|---|
| Hợp đồng và mã lõi | `src/contract/constants.py`, `nfe_tracker.py`, `base_optimizer.py` |
| Kiểm thử và CI | `tests/test_contract.py`, `scripts/check_test_isolation.sh`, `.github/workflows/ci.yml` |
| Tài liệu khởi động | `00_START_HERE/` (7 file ở 3.7) |
| Đăng ký trước | `00_START_HERE/PREREG.md` (commit sớm hơn mọi kết quả benchmark) |
| Nghiệm thu | Ghi chú nghiệm thu G1 trong PR của Thy và Nga; sha256 hai file mapping trong `CHANGE_REQUESTS.md` |
| PoC | `outputs/tables/poc_week1.csv`, slide báo cáo sơ bộ Tuần 1 |

## 5. Definition of Done (DoD) Tuần 1

- [ ] Repo hoạt động; `pytest tests/test_contract.py` đạt 13/13 PASS và CI xanh trên `main`.
- [ ] `scripts/check_test_isolation.sh` in `Cach ly Test: OK`; thử chèn `P_test` vào một file mẫu thì CI đỏ.
- [ ] `CONTRACT_v1.0.md` có chữ ký và ngày ban hành; CR-001…CR-005 ghi đủ; sha256 hai file mapping đã đóng băng.
- [ ] `PREREG.md` đã commit trước mọi kết quả benchmark (kiểm bằng `git log --format=%ct -- 00_START_HERE/PREREG.md`).
- [ ] Dữ liệu ViSoBERT thật qua script nghiệm thu: đúng shape, dtype, sha256; baseline Val ở ngưỡng 0.5 nằm trong ±0.5 điểm phần trăm so với 62.33%.
- [ ] PR của Thy và Nga đạt G1 và đã merge; không còn PR treo quá 24 giờ.
- [ ] Báo cáo sơ bộ với Giảng viên đã diễn ra, có hình địa hình hàm số và bảng so sánh Validation có số liệu thật.