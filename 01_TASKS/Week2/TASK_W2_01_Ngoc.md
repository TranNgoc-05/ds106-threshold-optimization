# TASK 01 — Ngọc — Tuần 2: Tích hợp 10 thuật toán, Audit NFE (G2) và Khóa siêu tham số (G3)

**Vai trò:** Technical Project Manager & System Architect  
**Owner:** Ngọc  
**Reviewers:** Thy (Core / Local), Nga (Global / Runner), Vinh (Surrogate / Eval)  
**Deadline tuần:** D+13 23:30  
**Mốc trong tuần:** M2 (smoke test 10 thuật toán trên dữ liệu thật) D+10 23:30 · Cổng G2 D+11 23:30 · M3 và Cổng G3 D+13 23:30

> **Quy ước thời gian.** D0 là ngày đầu Tuần 1; Tuần 2 gồm D+7 đến D+13.
>
> **Phân vai.** Thy: `coordinate_ascent`, `nelder_mead`, `powell`, landscape. Nga: `random_search`, `simulated_annealing`, `differential_evolution`, `pso`, `optuna_tpe`, `run_benchmark.py`, `validate_results.py`. Vinh: `surrogate.py`, `adam_surrogate`, `lbfgsb_surrogate`, `cv_lambda.py`.

## 1. Mục tiêu tuần (Weekly Goal)

1. Merge đủ 10 thuật toán vào `main`, tất cả kế thừa `BaseOptimizer`, không ghi đè `optimize()`, mọi lời gọi hàm mục tiêu đi qua tracker (Cổng G1).
2. Cổng G2: chạy `scripts/audit_nfe.py`, chênh lệch giữa bộ đếm độc lập và `actual_nfe` bằng **0** ở cả 3 budget cho cả 10 thuật toán.
3. Cổng G3: xác minh tuning công bằng (≤ 5 cấu hình, seeds 100–104, chỉ Validation, NFE=1000) và quy tắc 1-SE chọn `λ*`; ký đóng băng `hparams_frozen.yaml` và `lambda_star.json`.
4. Chốt kế hoạch tài nguyên cho mẻ benchmark 1801 lượt ở Tuần 3.

## 2. Input đầu vào

| Nguồn | Sản phẩm bàn giao | Đường dẫn |
|---|---|---|
| Thy | 3 thuật toán cục bộ; phân tích 200 điểm cực trị địa phương | `src/optimizers/local/`, `outputs/tables/landscape_summary.csv` |
| Nga | 5 thuật toán toàn cục và Bayes; `run_benchmark.py`; `validate_results.py`; profiling `ms/NFE` | `src/optimizers/global_/`, `src/runner/`, `reports/global_algorithms.md` |
| Vinh | `surrogate.py`, `tests/test_surrogate_gradcheck.py`, 2 thuật toán gradient; kết quả 5-fold CV | `src/optimizers/gradient/`, `outputs/tables/cv_lambda_results.csv`, `artifacts/cv_folds.npy` |
| Nga, Thy, Vinh | `tuning_log.csv` và cấu hình thắng của từng thuật toán | `outputs/tuning_log.csv`, `src/configs/hparams_frozen.yaml` (bản đề xuất) |

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Cổng G1 cho toàn bộ 10 thuật toán (D+7 → D+10)

**Bảng thuật toán, chủ sở hữu và điểm cần soi:**

| Thuật toán | Owner | Điểm soi riêng của Ngọc |
|---|---|---|
| `coordinate_ascent` | Thy | Chỉ chấp nhận khi `J` tăng chặt; mỗi lượt quét tốn `D·(K−1)` NFE; dừng khi quét đầy đủ ở `w ≤ w_min` không cải thiện |
| `nelder_mead` | Thy | `bounds`, `maxfev = tracker.remaining`, `fatol = 1e-5`; không nuốt lỗi khác ngoài `BudgetExhausted` |
| `powell` | Thy | `maxfev` của SciPy chỉ gần đúng; có test chứng minh `actual_nfe ≤ max_nfe` ở cả 3 budget |
| `random_search` | Nga | Dùng `rng` do `optimize()` cấp |
| `simulated_annealing` | Nga | Lịch nhiệt phụ thuộc budget; `ΔJ = 0` luôn được chấp nhận |
| `differential_evolution` | Nga | Tự cài, không dùng `scipy.optimize.differential_evolution`; phản xạ biên; chọn lọc `≥`; ghi `extra["diversity"]` |
| `pso` | Nga | Cập nhật đồng bộ theo thế hệ; kẹp vận tốc; ghi `extra["diversity"]` |
| `optuna_tpe` | Nga | `n_jobs=1`, `seed` cố định; `n_trials = tracker.remaining`; quy tắc K-5: nếu pilot D=28 vượt 10 phút thì `multivariate=False` |
| `adam_surrogate` | Vinh | Đánh giá rời rạc `x0` ngay đầu; reset moment mỗi giai đoạn `τ`; `best_score` luôn là `J_λ` rời rạc |
| `lbfgsb_surrogate` | Vinh | `maxfun = stage_budget − 1`; đánh giá rời rạc cuối mỗi giai đoạn |

**Script kiểm tra tự động `tests/test_registry_conformance.py`** (Ngọc commit, áp cho mọi PR thuật toán; yêu cầu `src/runner/registry.py` xuất `REGISTRY` theo CR-005):

```python
import inspect

import pytest

from src.contract.base_optimizer import BaseOptimizer
from src.contract.constants import ALGO_REGISTRY_NAMES
from src.runner.registry import REGISTRY

EXPECTED = sorted(set(ALGO_REGISTRY_NAMES) - {"constant_0.5"})


def test_registry_has_exactly_ten_algorithms():
    assert sorted(REGISTRY) == EXPECTED and len(EXPECTED) == 10


@pytest.mark.parametrize("name", EXPECTED)
def test_inherits_base_and_does_not_override_optimize(name):
    cls = REGISTRY[name]
    assert issubclass(cls, BaseOptimizer)
    assert cls.optimize is BaseOptimizer.optimize
    assert cls.name == name
    assert "_search" in cls.__dict__


@pytest.mark.parametrize("name", EXPECTED)
def test_module_does_not_touch_raw_objective_or_global_seed(name):
    source = inspect.getsource(inspect.getmodule(REGISTRY[name]))
    for forbidden in ("make_objective", "np.random.seed", "random.seed"):
        assert forbidden not in source, f"{name}: chứa '{forbidden}'"
```

**Kiểm tra sai phân số học độc lập của surrogate.** Ngọc không chỉ tin test của Vinh mà chạy script riêng (`scripts/check_gradient_independent.py`). Ngưỡng bắt buộc là sai số tương đối `< 1e-5`:

```python
import numpy as np
from src.optimizers.gradient.surrogate import make_surrogate
from src.core.objective import make_objective  # chỉ để lấy cùng dữ liệu; không dùng trong optimizer

P, Y = np.load("src/data/P_val.npy"), np.load("src/data/Y_val.npy")  # CHỈ Validation
t_g = 0.55  # thay bằng giá trị trong artifacts/t_g.json
h, rng, worst = 1e-6, np.random.default_rng(2026), 0.0
for mode, D in (("global", 1), ("freq3", 3), ("per_label", 28)):
    lam = 0.0 if D == 1 else 0.1
    g = make_surrogate(P, Y, mode, lam, t_g)
    for tau in (0.1, 0.03, 0.01):
        for _ in range(20):
            theta = rng.uniform(0.1, 0.9, size=D)
            _, analytic = g(theta, tau)
            numeric = np.empty(D)
            for j in range(D):
                e = np.zeros(D); e[j] = h
                numeric[j] = (g(theta + e, tau)[0] - g(theta - e, tau)[0]) / (2 * h)
            rel = np.linalg.norm(analytic - numeric) / max(np.linalg.norm(numeric), 1e-12)
            worst = max(worst, rel)
            assert rel < 1e-5, (mode, tau, rel)
print("Gradient check độc lập: PASS, sai số tương đối lớn nhất =", worst)
```

**Điều kiện merge (mọi PR):** checklist 5.1 của Task Package được tick đủ; `pytest` xanh; `bash scripts/check_test_isolation.sh` sạch; Vinh chỉ được merge `adam_surrogate` và `lbfgsb_surrogate` sau khi gradient check độc lập ở trên PASS.

**Yêu cầu giao diện của `validate_results.py` (Nga):** `python -m src.runner.validate_results --csv ... --theta-log ... --traj-dir ... --expected-rows N`, với `N = 1801` ở chế độ đầy đủ; ở chế độ smoke (`--expected-rows 91`) kiểm tra mọi ràng buộc còn lại.

### 3.2 Smoke test M2 trên dữ liệu thật (D+10)

Nga chạy 10 thuật toán × D ∈ {1, 3, 28} × seeds 0–2 × NFE=200:

```bash
python -m src.runner.run_benchmark --algos all --dims 1 3 28 --budgets 200 --seeds 0-2 \
  --lam 0.1 --tag pilot --n-jobs 4 --resume --out outputs/benchmark_results.csv
python -m src.runner.validate_results --csv outputs/benchmark_results_pilot.csv \
  --theta-log outputs/benchmark_theta_log.jsonl --traj-dir outputs/trajectories --expected-rows 91
```

Kết quả kỳ vọng là 91 dòng (10 × 3 × 3 + 1 dòng `constant_0.5`). Ngọc ghi lại thuật toán nào dừng sớm (`is_converged=True`) để chuẩn bị phân tích Chapter 6 ở Tuần 3.

### 3.3 Cổng G2: Audit NFE (D+11)

**Nguyên tắc.** Với mỗi thuật toán, bọc hàm mục tiêu (và surrogate) bằng bộ đếm **nằm ngoài** `NFETracker`. Chạy ở NFE ∈ {200, 1000, 5000}, D ∈ {1, 3, 28}. `powell` và `optuna_tpe` chạy 10 seeds, các thuật toán khác chạy seeds 0–2. So sánh số lời gọi thực với `actual_nfe` do tracker ghi.

**Script `scripts/audit_nfe.py`** (phần lõi `audit_run` và đối chứng âm đã được chạy thử; chế độ audit đầy đủ cần `REGISTRY`, `make_objective`, `make_surrogate` thật):

```python
#!/usr/bin/env python
"""Cổng G2 — Kiểm toán ngân sách NFE độc lập (Ngọc sở hữu).

Với mỗi thuật toán trong REGISTRY: bọc hàm mục tiêu (và surrogate) bằng bộ đếm độc lập
nằm NGOÀI NFETracker, chạy ở 3 budget chính thức, rồi đối chiếu với actual_nfe do tracker ghi.
Mọi chênh lệch khác 0 là vi phạm; script thoát với mã 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.contract.base_optimizer import BaseOptimizer
from src.contract.nfe_tracker import LO, HI, OFFICIAL_BUDGETS, NFETracker

MODE_OF_D = {1: "global", 3: "freq3", 28: "per_label"}
GRADIENT_ALGOS = {"adam_surrogate", "lbfgsb_surrogate"}
STRESS_ALGOS = {"powell", "optuna_tpe"}      # dễ gọi lọt NFE: chạy nhiều seed hơn
BASE_SEEDS = (0, 1, 2)
STRESS_SEEDS = tuple(range(10))


class IndependentCounter:
    """Bộ đếm lời gọi nằm ngoài bộ nhớ của thuật toán và của tracker."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def audit_run(opt: BaseOptimizer, D: int, seed: int, budget: int, objective_fn, surrogate_fn=None) -> dict:
    obj = IndependentCounter(objective_fn)
    sur = IndependentCounter(surrogate_fn) if surrogate_fn is not None else None
    tracker = NFETracker(obj, budget, surrogate=sur)
    _, _, traj = opt.optimize(tracker, seed)
    external = obj.calls + (sur.calls if sur is not None else 0)
    violations = []
    if external != traj["actual_nfe"]:
        violations.append(f"counter={external} != actual_nfe={traj['actual_nfe']}")
    if traj["actual_nfe"] != tracker.nfe:
        violations.append(f"actual_nfe={traj['actual_nfe']} != tracker.nfe={tracker.nfe}")
    if len(traj["best_so_far"]) != traj["actual_nfe"]:
        violations.append("len(best_so_far) != actual_nfe")
    if external > budget:
        violations.append(f"vuot budget: {external} > {budget}")
    if (not traj["is_converged"]) and traj["actual_nfe"] != budget:
        violations.append("is_converged=False nhung actual_nfe != budget")
    return {"algorithm_name": opt.name, "dimension_D": D, "seed": seed, "nfe_budget": budget,
            "external_calls": external, "actual_nfe": traj["actual_nfe"],
            "diff": external - traj["actual_nfe"], "is_converged": traj["is_converged"],
            "violations": "; ".join(violations)}


class CheatingOptimizer(BaseOptimizer):
    """Đối chứng âm: gọi hàm mục tiêu gốc vòng qua tracker. Audit PHẢI bắt được lỗi này."""
    name = "cheating_control"

    def _search(self, tracker, x0, rng):
        for _ in range(10):
            tracker(x0)
        tracker.objective(x0)  # lời gọi lọt, không được tracker đếm
        return True


def self_test() -> int:
    f = lambda th: -float(np.sum((th - 0.6) ** 2))
    res = audit_run(CheatingOptimizer(D=3), 3, 0, 200, f)
    caught = res["diff"] == 1 and "counter=11 != actual_nfe=10" in res["violations"]
    print("SELF-TEST:", "PASS (audit bắt được lời gọi lọt)" if caught else "FAIL", res)
    return 0 if caught else 1


def load_problem(args):
    from src.core.objective import make_objective                     # Thy sở hữu
    from src.optimizers.gradient.surrogate import make_surrogate      # Vinh sở hữu
    data = Path(args.data_dir)
    P, Y = np.load(data / "P_val.npy"), np.load(data / "Y_val.npy")    # CHỈ tập Validation
    t_g = args.t_g if args.t_g is not None else json.loads(Path(args.t_g_file).read_text())["t_g"]

    def build(D):
        mode = MODE_OF_D[D]
        lam = 0.0 if D == 1 else args.lam
        obj = make_objective(P, Y, mode, "argmax", True, lam, t_g)
        sur = make_surrogate(P, Y, mode, lam, t_g)
        return obj, sur
    return build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", default="src/data")
    ap.add_argument("--t-g", type=float, default=None)
    ap.add_argument("--t-g-file", default="artifacts/t_g.json")
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--hparams", default=None, help="YAML: {algorithm_name: {hyperparam: value}}")
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 3, 28])
    ap.add_argument("--budgets", type=int, nargs="+", default=list(OFFICIAL_BUDGETS))
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--compare-log", default=None, help="CSV benchmark/pilot để đối chiếu actual_nfe")
    ap.add_argument("--out", default="reports/audit_G2_nfe.md")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    from src.runner.registry import REGISTRY
    hp = {}
    if args.hparams:
        import yaml
        hp = yaml.safe_load(Path(args.hparams).read_text()) or {}
    build = load_problem(args)
    names = args.algos or [n for n in REGISTRY if n != "constant_0.5"]
    rows = []
    for name in names:
        for D in args.dims:
            obj, sur = build(D)
            seeds = STRESS_SEEDS if name in STRESS_ALGOS else BASE_SEEDS
            for budget in args.budgets:
                for seed in seeds:
                    opt = REGISTRY[name](D=D, **hp.get(name, {}))
                    rows.append(audit_run(opt, D, seed, budget, obj, sur if name in GRADIENT_ALGOS else None))
    df = pd.DataFrame(rows)

    if args.compare_log and Path(args.compare_log).exists():
        log = pd.read_csv(args.compare_log)
        key = ["algorithm_name", "dimension_D", "seed", "nfe_budget"]
        merged = df.merge(log[key + ["actual_nfe"]], on=key, how="inner", suffixes=("", "_log"))
        bad = merged[merged["external_calls"] != merged["actual_nfe_log"]]
        for _, r in bad.iterrows():
            df.loc[(df[key] == r[key]).all(axis=1), "violations"] += "; LOG_MISMATCH"

    summary = (df.groupby(["algorithm_name", "dimension_D", "nfe_budget"])
                 .agg(runs=("seed", "count"), max_abs_diff=("diff", lambda s: int(s.abs().max())),
                      converged=("is_converged", "sum"))
                 .reset_index())
    failed = df[df["violations"] != ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# Báo cáo nghiệm thu Cổng G2 — Audit NFE\n\n")
        fh.write(f"Tổng số lần chạy kiểm toán: {len(df)}. Số lần vi phạm: {len(failed)}.\n\n")
        fh.write(summary.to_markdown(index=False) + "\n\n")
        if len(failed):
            fh.write("## Vi phạm\n\n" + failed.to_markdown(index=False) + "\n")
    print(summary.to_string(index=False))
    print("G2:", "PASS" if failed.empty else f"FAIL ({len(failed)} vi phạm, xem {args.out})")
    return 0 if failed.empty else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Trình tự thực thi:**

```bash
python scripts/audit_nfe.py --self-test                # bắt buộc PASS trước khi tin kết quả audit
python scripts/audit_nfe.py --lam 0.1 --compare-log src/outputs/benchmark_results_pilot.csv
```

**Xử lý vi phạm:** thuật toán có `diff ≠ 0` bị trả PR cho owner; sau khi sửa, Ngọc chạy lại toàn bộ audit cho thuật toán đó ở cả 3 budget và cả 3 D. Hai trường hợp phải kiểm tra thủ công kỹ nhất: `powell` (SciPy có thể gọi quá `maxfev`) và `optuna_tpe` (số trial so với `tracker.remaining`). Audit cần chạy lại một lần sau khi đóng băng `hparams_frozen.yaml`, vì siêu tham số có thể thay đổi hành vi dừng sớm (`--hparams src/configs/hparams_frozen.yaml`).

### 3.4 Cổng G3: Tuning công bằng và đóng băng cấu hình (D+12 → D+13)

**Kiểm tra tuning (script `scripts/check_g3.py`, đã chạy thử trên dữ liệu giả; kiểm tra các điều kiện dưới đây và quy tắc 1-SE):**

```python
#!/usr/bin/env python
"""Cổng G3 — kiểm tra công bằng tuning và quy tắc 1-SE (Ngọc sở hữu)."""
import json, sys
from pathlib import Path
import pandas as pd

OK_SEEDS = set(range(100, 105))
errors = []

tl = pd.read_csv("outputs/tuning_log.csv")
need = {"algorithm", "config_id", "D", "seed", "nfe_budget", "split", "best_val_score"}
if not need <= set(tl.columns):
    errors.append(f"tuning_log thiếu cột: {sorted(need - set(tl.columns))}")
else:
    for algo, g in tl.groupby("algorithm"):
        if g["config_id"].nunique() > 5: errors.append(f"{algo}: {g['config_id'].nunique()} cấu hình > 5")
    if not set(tl["seed"]) <= OK_SEEDS: errors.append("Có seed ngoài 100–104")
    if set(tl["nfe_budget"]) != {1000}: errors.append("Tuning phải ở NFE=1000")
    if set(tl["split"]) != {"val"}: errors.append("Tuning chỉ được dùng split=val")
    if not set(tl["D"]) <= {3, 28}: errors.append("Tuning chỉ ở D ∈ {3, 28}")

cv = pd.read_csv("outputs/tables/cv_lambda_results.csv")  # cột: dimension_D, lam, mean_heldout_macro_f1, se
star = json.loads(Path("artifacts/lambda_star.json").read_text())
for D, g in cv.groupby("dimension_D"):
    best = g.loc[g["mean_heldout_macro_f1"].idxmax()]
    thr = best["mean_heldout_macro_f1"] - best["se"]
    expected = float(g[g["mean_heldout_macro_f1"] >= thr]["lam"].max())
    if abs(float(star[str(int(D))]) - expected) > 1e-12:
        errors.append(f"D={int(D)}: lambda_star={star[str(int(D))]} nhưng quy tắc 1-SE cho {expected}")

print("G3:", "PASS" if not errors else "FAIL"); [print(" -", e) for e in errors]
sys.exit(1 if errors else 0)
```

**Các kiểm tra bổ sung bằng tay:**

1. **Cấu hình thắng đúng quy tắc:** với từng thuật toán, cấu hình trong `hparams_frozen.yaml` là cấu hình có trung bình `best_val_score` cao nhất trên hai D ∈ {3, 28} (tính lại bằng pandas từ `tuning_log.csv`).
2. **Phạm vi hợp lệ của 5-fold CV:** `grep -n "test" src/eval/cv_lambda.py` không chứa mã nạp tập Test; fold sinh bởi `MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (hoặc `KFold` nếu thiếu thư viện, có ghi chú); `t_g` được tính lại trên 4 fold huấn luyện trong từng fold; `λ ∈ {0.0, 0.1, 0.5}`; seeds `100–104`; thuật toán dùng là `coordinate_ascent`, NFE=1000.
3. **Đủ 10 thuật toán** có cấu hình trong `hparams_frozen.yaml`; không có giá trị nào được sửa tay sau `tuning_log.csv`.

**Đóng băng.** Sau khi `check_g3.py` PASS:

```bash
sha256sum src/configs/hparams_frozen.yaml artifacts/lambda_star.json > src/artifacts/FROZEN_HASHES.txt
git add -A && git commit -m "G3: đóng băng hparams_frozen.yaml và lambda_star.json" && git tag g3-frozen
```

Từ thời điểm này, mọi thay đổi hai file cần Change Request. Mã băm được kiểm lại ở Tuần 3 (cùng với `LOCK_FREEZE.json`).

### 3.5 Kế hoạch tài nguyên cho Tuần 3

Từ profiling của Nga (`reports/global_algorithms.md`), Ngọc ghi vào `reports/audit_G2_nfe.md` (mục phụ lục): thời gian `ms/NFE` thực của từng thuật toán, ước lượng tổng thời gian cho 1801 lượt chạy, số lõi CPU, `--n-jobs`, thứ tự chạy ưu tiên thuật toán chậm (`optuna_tpe` ở D=28) trước. Nếu ước lượng vượt 12 giờ tường thì Ngọc quyết định phương án rút gọn trước M3 và ghi vào `CHANGE_REQUESTS.md`. Quyết định K-5 về `multivariate=False` cho D=28 đã được duyệt trước và không cần xin lại.

## 4. Output tuần

| Nhóm | File |
|---|---|
| Kiểm toán | `scripts/audit_nfe.py`, `reports/audit_G2_nfe.md` (kèm phụ lục kế hoạch tài nguyên) |
| Kiểm thử | `tests/test_registry_conformance.py`, `scripts/check_gradient_independent.py`, `scripts/check_g3.py` |
| Đóng băng | Chữ ký phê duyệt G3 trong `CHANGE_REQUESTS.md`; `src/configs/hparams_frozen.yaml`, `artifacts/lambda_star.json`, `src/artifacts/FROZEN_HASHES.txt`, tag `g3-frozen` |
| Báo cáo nội bộ | Báo cáo tiến độ nội bộ Tuần 2 (tình trạng 10 thuật toán, kết quả G2 và G3, rủi ro Tuần 3) |

## 5. Definition of Done (DoD) Tuần 2

- [ ] Đủ 10 thuật toán đã merge vào `main`; `pytest` xanh, gồm `test_registry_conformance.py`; `check_test_isolation.sh` sạch.
- [ ] Gradient check độc lập đạt sai số tương đối `< 1e-5` ở cả ba `mapping_mode` và ba mức `τ`.
- [ ] **Cổng G2:** `python scripts/audit_nfe.py --self-test` PASS và audit đầy đủ báo `max_abs_diff = 0` cho 10 thuật toán × 3 budget × 3 D; `reports/audit_G2_nfe.md` đã ký.
- [ ] **Cổng G3:** `scripts/check_g3.py` PASS; `hparams_frozen.yaml` có đủ 10 thuật toán; `lambda_star.json` có `λ*` cho D=3 và D=28 theo quy tắc 1-SE; hai file đã được băm và gắn tag `g3-frozen`.
- [ ] Smoke test M2 (91 dòng) qua `validate_results` ở chế độ smoke.
- [ ] Kế hoạch tài nguyên Tuần 3 đã chốt, ước lượng thời gian không vượt 12 giờ tường hoặc có phương án được ký.