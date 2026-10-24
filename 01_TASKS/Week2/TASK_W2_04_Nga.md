# TASK_04_Nga_Week2 — Profiling, Tuning (LD-7), Đóng băng siêu tham số & Cổng G2/G3

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh Global: đo hiệu năng thực, tinh chỉnh công bằng, hợp nhất cấu hình toàn dự án |
| **Owner** | Nga |
| **Reviewers** | Ngọc  (Leader, A: phê duyệt K-5, kiểm toán G2, ký G3), Vinh (Peer Reviewer: tái lập, runner) |
| **Deadline** | Cuối Tuần 2, tương ứng mốc M3 (đóng băng siêu tham số) và Cổng G2/G3 |
| **Phối hợp** | Thy (C: cấu hình thắng của Coordinate Ascent, Nelder-Mead, Powell), Vinh (C: cấu hình thắng của Adam, L-BFGS-B), Ngọc (A) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Đo `ms/NFE` thật trên dữ liệu thật cho từng thuật toán × D × budget, ước lượng tổng thời gian benchmark 1.801 lượt, kích hoạt quyết định K-5 cho TPE ở D=28 khi cần.
2. Tinh chỉnh siêu tham số theo LD-7 cho các thuật toán của Nga, ghi `outputs/tuning_log.csv`.
3. Hợp nhất cấu hình thắng của Nga, Thy và Vinh vào `configs/hparams_frozen.yaml`.
4. Soạn `reports/global_algorithms.md` (tối đa 3 trang).
5. Vượt Cổng G2 (audit NFE chênh lệch 0) và nộp Cổng G3 cho Ngọc ký.

---

## 2. Input đầu vào

- Thy: `P_val.npy`, `Y_val.npy` thật, `mapping_freq3.json`, `artifacts/t_g.json`, thuật toán cục bộ đã tích hợp registry.
- Vinh: thuật toán gradient (`adam_surrogate`, `lbfgsb_surrogate`) và log tuning của riêng Vinh. Lưu ý: nếu tới thời điểm profiling mà các thuật toán gradient chưa merge thì ghi rõ mục đó là "chưa đo" trong bảng profiling và đo lại ngay khi merge (trước Tuần 3).
- Ngọc: `scripts/audit_nfe.py`, hợp đồng 2.8/2.9.
- Tuần 1 của Nga: 5 thuật toán, runner, `configs/hparams_default.yaml`.
- **Quy ước λ khi tuning (Ngọc chốt):** cố định **λ = 0.1** cho cả D=3 và D=28 để giải phóng phụ thuộc vòng với 5-Fold CV chọn λ\* của Vinh. Ghi tường minh vào `tuning_log.csv` (cột `lambda`). Benchmark chính ở Tuần 3 mới dùng `lambda_star.json`.
- Tài liệu đọc trước: DS106 Chapter 2 (pp. 1–18); Kirkpatrick et al. (1983); Storn & Price (1997); Clerc & Kennedy (2002); Akiba et al. (2019); Bergstra et al. (2011).

**Quy tắc kỹ thuật áp dụng cả tuần**

- **LD-7:** mỗi thuật toán tối đa 5 cấu hình, chỉ trên tập Validation, NFE=1000, seeds `100–104`, D ∈ {3, 28}. Không dùng Test, không dùng seeds 0–19.
- Cấu hình bị loại phải được ghi **trước** khi chạy (tránh chọn sau khi thấy kết quả).
- Cách ly Test: không import, không nhắc `P_test`, `Y_test`, `_test.npy`.
- Mọi lượt đo/tuning dùng cùng runner và cùng cách tính `time_seconds`, tách khỏi file chính: ghi bằng `--tag` (không bao giờ vào `benchmark_results.csv`).

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Pilot K-5 cho Optuna TPE ở D=28 (làm đầu tuần)

- Chạy **1** lần TPE ở D=28, NFE=5000 với `multivariate=True`, `n_startup_trials=20`, seed 100 (chỉ để đo thời gian, kết quả không dùng làm bằng chứng khoa học):

  ```bash
  python -m src.runner.run_benchmark --algos optuna_tpe --dims 28 --budgets 5000 --seeds 100-100 \
    --lam 0.1 --hparams configs/tuning/tpe_mvT_ns20.yaml --config-id tpe_mvT_ns20 \
    --mapping auto --fallback argmax --exclusive-neutral true --n-jobs 1 --tag pilot
  ```
- **Quy tắc K-5 (đã được Ngọc duyệt trước, không cần xin lại):** nếu `time_seconds` > **600 giây (10 phút)** thì đặt `multivariate=False` làm **override riêng cho D=28** trong YAML. Tham số `n_startup_trials` vẫn lấy theo kết quả tuning.
- Ghi số đo (thời gian, số trial, thời gian trung bình mỗi trial) và quyết định vào `reports/global_algorithms.md`. Đo thêm `multivariate=False` ở D=28, B=5000. Nếu **cả** `multivariate=False` cũng vượt 10 phút thì quy tắc K-5 không còn đủ, báo Ngọc ngay (không tự quyết định).
- Đo thêm thời gian trung bình mỗi trial theo budget (200, 1000, 5000) và theo `multivariate`, cho `reports/global_algorithms.md`.

### 3.2. Profiling thời gian thực và ước lượng tải

- Với từng thuật toán đã tích hợp, từng D ∈ {1, 3, 28}, từng budget ∈ {200, 1000, 5000}: chạy 3 seeds (seeds 100–102), ghi `time_seconds` và tính `ms/NFE = 1000·time_seconds/actual_nfe`. Ghi `outputs/profiling_runtime.csv`: `algorithm, D, nfe_budget, n_runs, ms_per_nfe_mean, ms_per_nfe_std, time_seconds_mean`.
- Đo **hiệu suất song song** `η = T(1 tiến trình) / (n_jobs × T(n_jobs tiến trình))` bằng một mẻ nhỏ cùng nội dung chạy với `--n-jobs 1` và `--n-jobs 6`.
- Ước lượng tổng thời gian wall-clock của 1.801 lượt:

  ```text
  T_total ≈ Σ_{algo, D, budget} 20 · time_seconds_mean(algo, D, budget) / (n_jobs · η)
  ```
- Nếu `T_total` vượt **12 giờ** tường với số lõi hiện có, báo Ngọc để điều chỉnh `--n-jobs` hoặc kế hoạch. Kết quả (bảng chi phí, `T_total`, `η`, số lõi) vào `reports/global_algorithms.md`.
- Phụ lục tính tải cho các đợt sau để Ngọc phân bổ CPU: ~160 lượt áp lực chọn lọc, ~600 lượt `sep_check` của Vinh, ~400 lượt bổ sung và ~160 lượt ablation.

### 3.3. Tinh chỉnh siêu tham số (LD-7)

**Quy trình chạy:** mỗi cấu hình có một file YAML riêng trong `configs/tuning/`, chạy bằng runner với `--tag tuning --config-id <id> --hparams <yaml> --lam 0.1 --dims 3 28 --budgets 1000 --seeds 100-104`. Dữ liệu thô vào `outputs/benchmark_results_tuning.csv`. Một script chuyển đổi tạo `outputs/tuning_log.csv`.

**Lưới cấu hình (đã chốt):**

| Thuật toán | Cấu hình đánh giá | Số cấu hình |
|---|---|---|
| `random_search` | Không có siêu tham số, không tuning | 0 |
| `simulated_annealing` | `T0 ∈ {1e-3, 5e-3, 2e-2}` × `σ0 ∈ {0.05, 0.1}` (6 tổ hợp), **loại 1**. Loại `sa_T0_2e-2_s0.05` (nhiệt độ cao nhất với bước nhỏ nhất ít hợp lý nhất). Nga ghi lựa chọn loại và lý do **trước khi chạy** | 5 |
| `differential_evolution` | `{rand1, best1} × F ∈ {0.5, 0.8}`, `CR = 0.9` | 4 |
| `pso` | `topology ∈ {gbest, ring} × w ∈ {0.4, 0.729}` | 4 |
| `optuna_tpe` | `multivariate ∈ {False, True} × n_startup_trials ∈ {10, 20, 50}`, **loại** `(True, 50)` | 5 |

- Mã cấu hình: `sa_T0_<giá trị>_s<giá trị>`, `de_<strategy>_F<giá trị>`, `pso_<topology>_w<giá trị>`, `tpe_mv<F|T>_ns<giá trị>`.
- Số lượt tuning: SA 5×2×5 = 50; DE 4×2×5 = 40; PSO 4×2×5 = 40; TPE 5×2×5 = 50; tổng 180 lượt, NFE = 1000.

**`outputs/tuning_log.csv`** (5 cột đầu theo MD, cột bổ sung ở cuối): `algorithm, config_id, D, seed, best_val_score, lambda, status, note`.

- Dòng cấu hình đã đánh giá: `status=evaluated`, `lambda=0.1`.
- Dòng cấu hình bị loại: một dòng mỗi cấu hình, `status=excluded`, `D`/`seed`/`best_val_score` để trống, `note` nêu lý do. Dòng này được ghi trước khi bắt đầu chạy.
- Kiểm tra tự động: với mỗi thuật toán, số `config_id` có `status=evaluated` ≤ 5; seeds ⊂ {100..104}; D ⊂ {3, 28}; NFE=1000.

**Quy tắc chọn cấu hình thắng (LD-7):** với mỗi `config_id`, lấy trung bình `best_val_score` theo seeds trong từng D, rồi trung bình hai D. Cấu hình có trung bình cao nhất thắng. Hòa thì chọn cấu hình xuất hiện trước trong thứ tự lưới (mặc định ưu tiên cấu hình mặc định).

**Trường hợp K-5 đã kích hoạt:** cấu hình thắng của TPE chọn theo quy tắc chung. Nếu thắng có `multivariate=True` thì YAML ghi cấu hình thắng ở mức chung, cộng override `D=28: multivariate=False` với **cùng** `n_startup_trials`. Báo cáo phải nêu điểm tuning quan sát được của ô override (cấu hình `(False, ns)` với `ns` tương ứng đã có trong lưới) để người đọc thấy tác động của override.

### 3.4. Hợp nhất và đóng băng `configs/hparams_frozen.yaml` (Cổng G3)

Thu thập cấu hình thắng của Thy (Coordinate Ascent, Nelder-Mead, Powell) và Vinh (Adam, L-BFGS-B) từ log tuning của họ. Nga **không sửa** nội dung của họ, chỉ xác minh log chứng minh ≤ 5 cấu hình mỗi thuật toán và seeds 100–104. Lược đồ đề xuất:

```yaml
version: 1
tuning_protocol: {nfe: 1000, seeds: [100, 101, 102, 103, 104], D: [3, 28], lambda_convention: 0.1}
algorithms:
  random_search: {}
  simulated_annealing: {T0: <thắng>, sigma0: <thắng>, T_end: 1.0e-5, sigma_end: 0.005}
  differential_evolution: {strategy: <thắng>, F: <thắng>, CR: 0.9}
  pso: {topology: <thắng>, w: <thắng>, c1: 1.49445, c2: 1.49445}
  optuna_tpe:
    multivariate: <thắng>
    n_startup_trials: <thắng>
    overrides: {"28": {multivariate: false}}     # chỉ khi K-5 kích hoạt
  coordinate_ascent: {...}   # của Thy, giữ nguyên
  nelder_mead: {...}         # của Thy
  powell: {...}              # của Thy
  adam_surrogate: {...}      # của Vinh
  lbfgsb_surrogate: {...}    # của Vinh
```

- File phải có cấu hình cho cả **10** thuật toán. Runner hỗ trợ `overrides` theo D.
- Nga nộp cho Ngọc ký (Cổng G3). Sau khi ký, file không đổi. Ghi SHA-256 của file vào nhật ký mỗi lần chạy chính.

### 3.5. Báo cáo `reports/global_algorithms.md` (tối đa 3 trang)

Cấu trúc:

1. Bảng tham số cuối cùng của 5 thuật toán (giá trị thắng và nguồn trong `tuning_log.csv`).
2. Quyết định thiết kế: SA chấp nhận khi `ΔJ=0` (di chuyển tự do trên plateau); DE chọn lọc `≥`, cập nhật đồng bộ, biên phản xạ; PSO đảo dấu vận tốc khi phản xạ, cập nhật `pBest` bằng `≥`, topology.
3. Pilot K-5 của TPE: số đo, quyết định, điểm tuning của ô override.
4. Chi phí thời gian thực so với NFE: bảng `ms/NFE`, thời gian trung bình mỗi trial của TPE theo budget, hiệu suất song song `η`, `T_total`.
5. Lịch phân bổ CPU đề xuất cho Tuần 3–4.

Báo cáo ghi nhận số liệu quan sát được, không giả định trước thuật toán nào thắng.

### 3.6. Nghiệm thu Cổng G2 và kiểm tra tái lập trên dữ liệu thật

- **Audit NFE:** `tests/test_global_optimizers.py` chạy lại trên dữ liệu thật: 5 thuật toán × budget {200, 1000, 5000} × D {1, 3, 28} (45 ca), bộ đếm độc lập bằng `tracker.nfe` bằng `actual_nfe` bằng budget. Cung cấp mã để Ngọc chạy `scripts/audit_nfe.py`, mục tiêu chênh lệch NFE bằng **0** tuyệt đối.
- **Tái lập:** chạy lại 10 khóa ngẫu nhiên cho `best_val_score` giống từng bit; `--n-jobs 1` và `--n-jobs 4` cho cùng kết quả. Lưu log vào `03_SUBMISSIONS/Nga/test_logs/reproducibility_test_log.txt`.
- **Khóa bảo vệ:** chạy lại test để chắc chắn runner từ chối ghi đè file chính từ các lượt `--tag tuning`/`--tag pilot`.

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Cấu hình | `configs/tuning/*.yaml`, `configs/hparams_frozen.yaml` (Ngọc ký) |
| Log, dữ liệu | `outputs/tuning_log.csv`, `outputs/benchmark_results_tuning.csv`, `outputs/profiling_runtime.csv`, `outputs/run_log.txt` |
| Báo cáo | `reports/global_algorithms.md` (kèm kết quả profiling và quyết định TPE ở D=28) |
| Test log | `03_SUBMISSIONS/Nga/test_logs/reproducibility_test_log.txt` |

---

## 5. Definition of Done (DoD tuần)

- [ ] Cổng G2: `audit_nfe.py` chênh lệch NFE bằng 0 trên cả 5 thuật toán của Nga ở 3 budget × 3 D.
- [ ] Pilot K-5 đã chạy, quyết định được ghi (kể cả khi không kích hoạt), `multivariate=False` chỉ là override riêng cho D=28 nếu kích hoạt.
- [ ] Profiling hoàn tất, `T_total` ước lượng và đã báo Ngọc nếu vượt 12 giờ.
- [ ] `tuning_log.csv` chứng minh ≤ 5 cấu hình mỗi thuật toán, seeds 100–104, D ∈ {3, 28}, NFE=1000, λ=0.1; cấu hình bị loại được ghi trước khi chạy.
- [ ] `hparams_frozen.yaml` có cấu hình cho cả 10 thuật toán, Ngọc ký duyệt (Cổng G3).
- [ ] Tái lập: 10 khóa ngẫu nhiên giống từng bit; `--n-jobs 1` và `--n-jobs 4` cho cùng kết quả; log đã lưu.
- [ ] `reports/global_algorithms.md` ≤ 3 trang, đủ năm phần.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py`.
- [ ] Không có lượt tuning/pilot nào ghi vào `benchmark_results.csv` hoặc `benchmark_theta_log.jsonl`.