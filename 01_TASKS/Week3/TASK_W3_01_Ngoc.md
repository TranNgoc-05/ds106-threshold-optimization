# TASK 01 — Ngọc — Tuần 3: Giám sát benchmark 1801 lượt, Cổng G4, niêm phong Test và phân tích hội tụ sớm

**Vai trò:** Technical Project Manager & System Architect  
**Owner:** Ngọc  
**Reviewers:** Thy (Core / Local), Nga (Global / Runner), Vinh (Surrogate / Eval)  
**Deadline tuần:** D+20 23:30  
**Mốc trong tuần:** Khởi chạy benchmark chính D+14 · Nghiệm thu giữa kỳ D+17 · M4 (đủ 1801 dòng) và Cổng G4 D+19 23:30 · Ban hành `LOCK_FREEZE.json` D+20 12:00 · Khung báo cáo D+20 23:30

> **Quy ước thời gian.** D0 là ngày đầu Tuần 1; Tuần 3 gồm D+14 đến D+20.
>
> **Phân vai.** Nga chạy benchmark chính, phụ trách `run_benchmark.py`, `validate_results.py` và nghiên cứu áp lực chọn lọc. Thy cung cấp landscape và Bảng 4. Vinh cung cấp `separable_oracle.py` và tính `optimality_gap_D28.csv`.

## 1. Mục tiêu tuần (Weekly Goal)

1. Giám sát mẻ chạy 1801 lượt (10 thuật toán × 3 D × 20 seeds × 3 budget + 1 dòng `constant_0.5`) do Nga thực hiện, đảm bảo kết quả không bị chỉnh tay và tái lập được.
2. Cổng G4: nghiệm thu `benchmark_results.csv` bằng `validate_results.py`, kiểm tra độc lập và nhân bản 10 khóa ngẫu nhiên.
3. Ký mã băm SHA-256 và ban hành `artifacts/LOCK_FREEZE.json`, mở cổng cho `test_gate.py` đọc tập Test thật ở Tuần 4.
4. Phân tích hội tụ sớm theo Chapter 6 *Premature Convergence* (pp. 1–9).
5. Chấp bút khung lý thuyết của báo cáo toàn văn.

## 2. Input đầu vào

| Nguồn | Sản phẩm | Đường dẫn |
|---|---|---|
| Nga | Kết quả benchmark chính, sidecar, quỹ đạo, log chạy | `outputs/benchmark_results.csv`, `benchmark_theta_log.jsonl`, `trajectories/*.npz`, `run_log.txt`, `optimality_gap_D28.csv`|
| Nga | Khung Bảng 1 (mean ± std, thời gian) | `src/outputs/tables/table1_val_performance.csv` |
| Nga | Nghiên cứu áp lực chọn lọc: DE (`rand1` / `best1`, `F ∈ {0.5, 0.8}`), PSO (`gbest` / `ring`, `w ∈ {0.4, 0.729}`) | `src/outputs/tables/selective_pressure_study.csv`, `figures/premature_convergence_de_pso.png` |
| Nga | Biểu đồ hội tụ | `figures/convergence_curves_D{1,3,28}.png`, `nfe_to_99pct_boxplot.png`, `runtime_vs_nfe.png` |
| Vinh | Gap tới nghiệm tối ưu chính xác trong cấu hình tách biến | `src/outputs/tables/optimality_gap_D28.csv` (cần batch `sep_check` do Nga chạy, xem 3.1) |
| Thy | Tóm tắt landscape | `src/outputs/tables/landscape_summary.csv`, biểu đồ 1D/2D |
| Hợp đồng | `hparams_frozen.yaml`, `lambda_star.json`, `FROZEN_HASHES.txt`, tag `g3-frozen` | `src/configs/`, `src/artifacts/` |

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Giám sát mẻ chạy chính (D+14 → D+19)

**Điều kiện khởi chạy** (Ngọc xác nhận trước khi Nga bấm chạy):

```bash
git status --porcelain                 # phải rỗng: không có thay đổi chưa commit
git describe --tags --exact-match HEAD # phải ra g3-frozen hoặc tag sau đó
sha256sum -c src/artifacts/FROZEN_HASHES.txt   # hparams và lambda_star chưa bị đổi
bash scripts/check_test_isolation.sh && pytest -q
```

**Lệnh chạy chính** (một lần duy nhất, không chỉnh tay file kết quả):

```bash
python -m src.runner.run_benchmark --algos all --dims 1 3 28 --budgets 200 1000 5000 --seeds 0-19 \
  --lam-file artifacts/lambda_star.json --hparams src/configs/hparams_frozen.yaml \
  --mapping auto --fallback argmax --exclusive-neutral true --n-jobs 6 --resume \
  --out outputs/benchmark_results.csv
```

**Nhịp giám sát hằng ngày** (ghi vào `reports/benchmark_monitor.md`): số dòng hiện có so với 1801; ETA từ `run_log.txt`; danh sách job lỗi (không được điền giá trị giả); không có hai tiến trình cùng ghi file (Nga là writer duy nhất theo thiết kế); `--resume` sau khi bị ngắt không sinh dòng trùng.

**Batch `sep_check` (D+17 → D+19), cần cho Optimality Gap.** Nga chạy 10 thuật toán ở D=28, NFE ∈ {200, 1000, 5000}, 20 seeds, trong cấu hình tách biến thuần, ghi ra file riêng, không trộn vào file chính:

```bash
python -m src.runner.run_benchmark --algos all --dims 28 --budgets 200 1000 5000 --seeds 0-19 \
  --lam-file artifacts/lambda_star.json --hparams src/configs/hparams_frozen.yaml \
  --mapping per_label --fallback none --exclusive-neutral false --n-jobs 6 --resume \
  --tag sep_check --out outputs/benchmark_results.csv
```

Vinh dùng kết quả này và `separable_oracle.py` để tính `optimality_gap_D28.csv`. Ngọc kiểm: không có `optimality_gap < −1e-9` (nếu có thì tracker hoặc objective sai, phải dừng và báo ngay); `separable_oracle` khớp brute-force lưới 0.0005 trên 3 nhãn trong sai số `1e-9`.

### 3.2 Cổng G4: nghiệm thu dữ liệu benchmark (D+19)

**Bước 1. Chạy `validate_results.py` của Nga ở chế độ đầy đủ:**

```bash
python -m src.runner.validate_results --csv src/outputs/benchmark_results.csv \
  --theta-log src/outputs/benchmark_theta_log.jsonl --traj-dir src/outputs/trajectories --expected-rows 1801
```

**Bước 2. Kiểm tra độc lập bằng script riêng của Ngọc** (không dùng lại mã của Nga), `scripts/check_g4_independent.py`:

```python
import json
import sys
from pathlib import Path

import pandas as pd

from src.contract.constants import ALGO_REGISTRY_NAMES, CSV_COLUMNS, EXPECTED_ROWS_FULL

df = pd.read_csv("outputs/benchmark_results.csv")
errors = []
if tuple(df.columns) != CSV_COLUMNS: errors.append(f"Sai header/thứ tự cột: {list(df.columns)}")
if len(df) != EXPECTED_ROWS_FULL: errors.append(f"Số dòng {len(df)} != {EXPECTED_ROWS_FULL}")
key = ["algorithm_name", "dimension_D", "seed", "nfe_budget"]
if df.duplicated(key).any(): errors.append("Có khóa chính trùng")
if not set(df["algorithm_name"]) <= set(ALGO_REGISTRY_NAMES): errors.append("Tên thuật toán ngoài registry")
body = df[df["algorithm_name"] != "constant_0.5"]
if len(body) != 1800: errors.append(f"Phần thân có {len(body)} dòng, cần 1800")
if not set(body["dimension_D"]) == {1, 3, 28}: errors.append("D ngoài {1,3,28}")
if not set(body["seed"]) == set(range(20)): errors.append("Seeds không phải 0..19")
if not set(body["nfe_budget"]) == {200, 1000, 5000}: errors.append("Budget ngoài {200,1000,5000}")
if (df["actual_nfe"] < 1).any() or (df["actual_nfe"] > df["nfe_budget"]).any(): errors.append("actual_nfe ngoài [1, budget]")
nc = df[~df["is_converged"].astype(bool)]
if (nc["actual_nfe"] != nc["nfe_budget"]).any(): errors.append("is_converged=False nhưng actual_nfe != budget")
const = df[df["algorithm_name"] == "constant_0.5"]
if len(const) != 1 or int(const.iloc[0]["nfe_budget"]) != 1 or int(const.iloc[0]["actual_nfe"]) != 1:
    errors.append("constant_0.5 phải là 1 dòng với nfe_budget=1, actual_nfe=1")
if df[["best_val_score", "time_seconds"]].isna().any().any(): errors.append("Có giá trị thiếu")

theta_keys = set()
for line in Path("outputs/benchmark_theta_log.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    theta_keys.add((r["algorithm_name"], r["dimension_D"], r["seed"], r["nfe_budget"]))
csv_keys = set(map(tuple, df[key].itertuples(index=False, name=None)))
if csv_keys != theta_keys: errors.append(f"Lệch khóa CSV/θ-log: {len(csv_keys ^ theta_keys)}")
for a, d, s, n in csv_keys:
    if not Path(f"outputs/trajectories/{a}_D{d}_s{s}_n{n}.npz").exists():
        errors.append(f"Thiếu quỹ đạo {a}_D{d}_s{s}_n{n}.npz"); break

print("G4 độc lập:", "PASS" if not errors else "FAIL"); [print(" -", e) for e in errors]
sys.exit(1 if errors else 0)
```

**Bước 3. Nhân bản 10 khóa ngẫu nhiên** (`scripts/replicate_keys.py`; phải khớp từng bit; luôn gồm ít nhất một khóa của `powell` và một của `optuna_tpe`):

```python
import subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

df = pd.read_csv("outputs/benchmark_results.csv")
body = df[df["algorithm_name"] != "constant_0.5"]
picks = pd.concat([body.sample(8, random_state=106),
                   body[body["algorithm_name"] == "powell"].sample(1, random_state=1),
                   body[body["algorithm_name"] == "optuna_tpe"].sample(1, random_state=2)])
bad = 0
for _, r in picks.iterrows():
    out = Path(tempfile.mkdtemp()) / "replica.csv"
    subprocess.run([sys.executable, "-m", "src.runner.run_benchmark", "--algos", r["algorithm_name"],
                    "--dims", str(r["dimension_D"]), "--budgets", str(r["nfe_budget"]),
                    "--seeds", f"{r['seed']}-{r['seed']}", "--lam-file", "artifacts/lambda_star.json",
                    "--hparams", "src/configs/hparams_frozen.yaml", "--n-jobs", "1", "--out", str(out)], check=True)
    rep = pd.read_csv(out).iloc[0]
    same = repr(float(rep["best_val_score"])) == repr(float(r["best_val_score"])) and int(rep["actual_nfe"]) == int(r["actual_nfe"])
    bad += not same
    print(r["algorithm_name"], int(r["dimension_D"]), int(r["seed"]), int(r["nfe_budget"]), "KHỚP" if same else "LỆCH")
sys.exit(1 if bad else 0)
```

**Bước 4. Kiểm tra thứ tự thời gian của đăng ký trước:**

```bash
git log --diff-filter=A --format=%ct -- 00_START_HERE/PREREG.md | tail -1
git log --diff-filter=A --format=%ct -- src/outputs/benchmark_results.csv | tail -1
# số thứ nhất phải nhỏ hơn số thứ hai
```

**Điều kiện đạt G4:** `validate_results` PASS, `check_g4_independent.py` PASS, 10/10 khóa nhân bản khớp từng bit, `sha256sum -c src/artifacts/FROZEN_HASHES.txt` vẫn khớp, thứ tự thời gian PREREG đúng. Bất kỳ điều kiện nào trượt thì trả về Nga kèm danh sách khóa lỗi; Ngọc không tự sửa dữ liệu.

Soạn `reports/audit_G4_acceptance.md` gồm: kết quả từng bước, số dòng, danh sách job lỗi (nếu có), tổng thời gian chạy, mã commit, và chữ ký.

### 3.3 Ký niêm phong SHA-256 và ban hành `LOCK_FREEZE.json` (D+20 sáng)

Chỉ chạy sau khi G4 đạt. Script `scripts/make_lock_freeze.py` (đã chạy thử, từ chối ghi đè khi thiếu `--force`):

```python
#!/usr/bin/env python
"""Cổng G4 — Ban hành artifacts/LOCK_FREEZE.json (chỉ Ngọc chạy, sau khi validate_results PASS)."""
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

FILES = ("outputs/benchmark_results.csv", "src/outputs/benchmark_theta_log.jsonl")
EXTRA = ("src/configs/hparams_frozen.yaml", "artifacts/lambda_star.json")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/LOCK_FREEZE.json")
    ap.add_argument("--force", action="store_true", help="cho phép ghi đè (cần Change Request)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit("LOCK_FREEZE.json đã tồn tại; không ghi đè nếu thiếu Change Request.")
    for p in FILES + EXTRA:
        if not Path(p).exists():
            sys.exit(f"Thiếu file bắt buộc: {p}")
    n_rows = sum(1 for _ in open(FILES[0], encoding="utf-8")) - 1
    if n_rows != 1801:
        sys.exit(f"benchmark_results.csv có {n_rows} dòng dữ liệu, cần 1801.")
    lock = {
        "contract_version": "1.0",
        "created_by": "Ngọc",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "files": {p: {"sha256": sha256_file(p)} for p in FILES},
        "frozen_configs": {p: {"sha256": sha256_file(p)} for p in EXTRA},
        "benchmark_rows": n_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã ghi", out)


if __name__ == "__main__":
    main()
```

**Script kiểm tra độc lập `scripts/verify_lock_freeze.py`** (thoát với mã 1 nếu có file lệch hash):

```python
import json, sys
from pathlib import Path
from scripts.make_lock_freeze import sha256_file

lock = json.loads(Path("artifacts/LOCK_FREEZE.json").read_text(encoding="utf-8"))
bad = [p for sect in ("files", "frozen_configs") for p, v in lock[sect].items()
       if sha256_file(p) != v["sha256"]]
print("LOCK_FREEZE:", "KHỚP" if not bad else f"LỆCH {bad}")
sys.exit(1 if bad else 0)
```

```bash
python scripts/make_lock_freeze.py
python scripts/verify_lock_freeze.py
git add artifacts/LOCK_FREEZE.json reports/audit_G4_acceptance.md
git commit -m "G4: niêm phong benchmark bằng SHA-256, mở cổng Test" && git tag g4-freeze
```

**Kiểm thử cơ chế mở cổng (bằng chứng cho DoD)**, trên một bản sao tạm của repo. Giả định `load_test()` gọi được không tham số theo hợp đồng 2.12:

```bash
rm -rf /tmp/gate_check && git clone . /tmp/gate_check && cd /tmp/gate_check
cp ../$(basename "$OLDPWD")/src/outputs/benchmark_results.csv src/outputs/ 2>/dev/null || cp "$OLDPWD"/src/outputs/benchmark_results.csv src/outputs/
cp "$OLDPWD"/src/outputs/benchmark_theta_log.jsonl src/outputs/ && cp -r "$OLDPWD"/src/data src/
python -c "from src.eval.test_gate import load_test; r = load_test(); print('Mở cổng khi nguyên vẹn: OK')"
python - <<'EOF'
p = "outputs/benchmark_results.csv"
raw = bytearray(open(p, "rb").read()); raw[-2] = (raw[-2] + 1) % 256
open(p, "wb").write(bytes(raw))
EOF
python -c "from src.eval.test_gate import load_test; load_test()" && echo "LỖI: cổng vẫn mở" || echo "Cổng từ chối khi lệch 1 byte: ĐẠT"
```

Ngọc chỉ xác nhận cổng mở hoặc đóng, **không** xem giá trị Test. Kết quả được dán vào `reports/audit_G4_acceptance.md`. Vinh cũng có `tests/test_test_gate.py` chứa ba ca: thiếu file khóa, lệch hash, chạy `--use_mock`.

### 3.4 Phân tích hội tụ sớm (Chapter 6 *Premature Convergence*, pp. 1–9)

Phần này là phân tích khám phá, không dùng để đổi cấu hình benchmark (xem `PREREG.md`).

**A. Định lượng hội tụ sớm của nhóm local và gradient** từ file benchmark chính:

```python
import pandas as pd

df = pd.read_csv("src/outputs/benchmark_results.csv")
body = df[df["algorithm_name"] != "constant_0.5"].copy()
body["budget_used"] = body["actual_nfe"] / body["nfe_budget"]
tab = (body.groupby(["algorithm_name", "dimension_D", "nfe_budget"])
           .agg(converged_rate=("is_converged", "mean"), budget_used=("budget_used", "mean"),
                median_score=("best_val_score", "median"))
           .round(4).reset_index())
tab.to_csv("outputs/tables/premature_convergence_local.csv", index=False)
print(tab[tab["converged_rate"] > 0].to_markdown(index=False))
```

Cách đọc: một thuật toán có `converged_rate` cao và `budget_used` thấp ở NFE=5000, đồng thời `median_score` thấp hơn nhóm toàn cục, là bằng chứng của hội tụ sớm (dừng tại điểm ổn định dưới mức tối ưu). Với `nelder_mead` cần đối chiếu giả thuyết simplex dẹt trên hàm hằng từng khúc.

**B. Áp lực chọn lọc của DE và PSO** từ `selective_pressure_study.csv`: so trung vị `diversity_collapse_nfe` giữa biến thể tham lam (`best1`, `gbest`, `w` nhỏ) và biến thể thăm dò (`rand1`, `ring`); đối soát với `best_val_score`. Kiểm tra: có đủ 8 biến thể (4 DE, 4 PSO) × 20 seeds; ô nào `diversity_collapse_nfe = None` thì ghi là không sụp đổ, không điền số. Kết luận phải có số cụ thể (ví dụ trung vị NFE sụp đổ của từng biến thể) kể cả khi trái với dự đoán "tham lam thì sụp đổ sớm hơn".

**C. Đối chiếu đường cong:** `convergence_curves_D{1,3,28}.png` có đủ dải 25–75 qua 20 seeds, thang NFE log, đủ 3 budget và mọi thuật toán. Ngọc trả lại các biểu đồ thiếu chú giải hoặc thang đo.

**D. Liên kết với giả thuyết:** ghi ba nhận xét vào `reports/chapter6_analysis.md`, mỗi nhận xét trỏ vào một dòng của bảng: (i) hội tụ sớm có xảy ra ở D=28 hay không; (ii) liên hệ với H3 (thuật toán tọa độ so với quần thể); (iii) liên hệ với H4 (L-BFGS-B hội tụ rất sớm có phải là bất lợi hay không).

### 3.5 Chấp bút khung lý thuyết của báo cáo toàn văn

Ngọc viết trước các chương nền tảng, đặt trong `reports/DS106_final_report.md` (chuyển sang `.docx` hoặc PDF ở Tuần 4):

| Phần | Nội dung | Căn cứ DS106 |
|---|---|---|
| 1. Đặt vấn đề | Phân loại đa nhãn mất cân bằng đuôi dài trên ViGoEmotions (20.664 bình luận, 28 nhãn gồm 27 nhãn sắc thái và Neutral); hạn chế của ngưỡng cố định 0.5 | Chapter 1 *Function Optimization* (pp. 1–10) |
| 2. Bối cảnh | Tối ưu ngưỡng là tối ưu hộp đen liên tục sau khi cố định ViSoBERT; cache `P_val`, `P_test`, 1÷3 ms mỗi lượt gọi | Chapter 2 *Optimization and Machine Learning* (pp. 1–18) |
| 3. Cơ sở lý thuyết | Không gian tìm kiếm, hàm mục tiêu, response surface hằng từng khúc; cực trị địa phương và toàn cục; chọn thuật toán | Chapter 1, 3 *Optimization Algorithm Selection* (pp. 1–15), 5 *Local and Global Optimization* (pp. 1–12) |
| 4. NFLT và baseline | Phát biểu Wolpert & Macready (1997), vì sao Random Search là baseline bắt buộc | Chapter 4 *No Free Lunch Theorem* (pp. 1–8) |
| 5. Hội tụ sớm | Hội tụ sớm và áp lực chọn lọc | Chapter 6 *Premature Convergence* (pp. 1–9) |
| 6. Mô hình hóa | `Θ = [0.05, 0.95]^D`, `D ∈ {1, 3, 28}`; `J_λ` chuẩn hóa theo D; quy tắc Neutral độc quyền theo so sánh xác suất chặt; Fallback; surrogate với `τ` từ 0.1 xuống 0.01 | Chapter 3 |
| 7. Giao thức thực nghiệm | NFE cố định {200, 1000, 5000}, 20 seeds, quy tắc đếm NFE (LD-2), tuning ≤ 5 cấu hình trên seeds 100–104, cô lập Test bằng `LOCK_FREEZE.json`, MWU-Holm | Chapter 4, 5 |

Quy tắc viết: ba câu chữ bắt buộc ở Tuần 1 (mục 3.5) được dùng nguyên văn; mọi phát biểu về "coupling" phải quy về Neutral độc quyền và Fallback; báo cáo chỉ nhắc ViSoBERT là mô hình duy nhất.

## 4. Output tuần

| Nhóm | File |
|---|---|
| Nghiệm thu G4 | `reports/audit_G4_acceptance.md`, `scripts/check_g4_independent.py`, `scripts/replicate_keys.py`, `reports/benchmark_monitor.md` |
| Niêm phong | `scripts/make_lock_freeze.py`, `scripts/verify_lock_freeze.py`, `src/artifacts/LOCK_FREEZE.json`, tag `g4-freeze` |
| Phân tích Chapter 6 | `reports/chapter6_analysis.md`, `outputs/tables/premature_convergence_local.csv` |
| Báo cáo | `reports/DS106_final_report.md` (hoàn thành các phần 1–7 ở mục 3.5) |

## 5. Definition of Done (DoD) Tuần 3

- [ ] **Cổng G4:** `benchmark_results.csv` đạt 1801 dòng hợp lệ (đúng 8 cột, không trùng khóa, `actual_nfe` đúng ràng buộc, đủ sidecar và quỹ đạo); `validate_results` và `check_g4_independent.py` đều PASS; 10/10 khóa nhân bản khớp từng bit.
- [ ] `artifacts/LOCK_FREEZE.json` đã tạo; `verify_lock_freeze.py` báo KHỚP; thử lệch 1 byte thì `test_gate.load_test()` **từ chối** (có log trong `audit_G4_acceptance.md`).
- [ ] `optimality_gap_D28.csv` không có giá trị `< −1e-9`; `separable_oracle` khớp brute-force.
- [ ] Biểu đồ hội tụ, nghiên cứu áp lực chọn lọc (đủ 8 biến thể × 20 seeds) và bảng `premature_convergence_local.csv` sẵn sàng cho phân tích Tuần 4.
- [ ] Khung báo cáo hoàn thành các phần Đặt vấn đề, Bối cảnh, Cơ sở lý thuyết (gồm đủ Chapter 1–6) và Mô hình hóa.
- [ ] `PREREG.md` có dấu thời gian sớm hơn commit của file benchmark chính.