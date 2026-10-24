# TASK_04_Nga_Week1 — Năm thuật toán toàn cục, khung Batch Runner & pilot báo cáo sơ bộ

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh Global: thuật toán tìm kiếm toàn cục / ngẫu nhiên / Bayes, baseline NFLT và hệ thống Batch Runner (DS106 — ViGoEmotions) |
| **Owner** | Nga |
| **Reviewers** | Ngọc (Leader, A: duyệt chuẩn `BaseOptimizer`, nghiệm thu), Vinh (Peer Reviewer: kiểm tra runner và tính tái lập) |
| **Deadline** | Cuối Tuần 1, bao gồm nội dung mốc M1 (RS, SA, DE, PSO trên mock) và M2 (TPE, runner, smoke test dữ liệu thật). Ngọc chốt TPE cài ngay trong Tuần 1. Pilot bàn giao vào Ngày 4–5 |
| **Phối hợp** | Thy (C: mock data, `P_val`, `Y_val`, `t_g`, `make_objective`), Ngọc (C: số liệu pilot cho slide), Vinh (người dùng các cờ CLI cho ablation và `sep_check`) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Cài đặt đủ 5 thuật toán `random_search`, `simulated_annealing`, `differential_evolution`, `pso`, `optuna_tpe`, kế thừa `BaseOptimizer`, chỉ cài `_search`, chạy hết ngân sách.
2. Xây dựng khung Batch Runner (`registry.py`, `run_benchmark.py`, `validate_results.py`) theo kiến trúc Single Writer, có `--resume`, ghi `benchmark_theta_log.jsonl` và quỹ đạo `.npz`, xử lý `constant_0.5`.
3. Viết `tests/test_global_optimizers.py` và `tests/test_runner.py`, chạy xanh trên `mock_fullshape`.
4. Khi Thy bàn giao `P_val`, `Y_val`, `t_g` thật, chạy pilot 3–5 seeds cho `random_search` và `optuna_tpe` tại D=3, NFE ∈ {200, 1000}, bàn giao cho Ngọc làm slide báo cáo tiến độ Tuần 1.

---

## 2. Input đầu vào

**Từ các thành viên khác**

- Ngọc: `src/contract/constants.py` (`LO=0.05`, `HI=0.95`), `nfe_tracker.py` (`NFETracker`, `BudgetExhausted`), `base_optimizer.py`, registry và schema theo hợp đồng mục 2.8 và 2.9.
- Thy: `02_SAMPLE_DATA/mock_small/` (smoke test shape/dtype), `mock_fullshape/` (unit test), `src/core/decision.py`, `objective.py` (`make_objective`), `data_io.py`, `artifacts/t_g.json` và `P_val.npy`, `Y_val.npy` thật (dự kiến Ngày 4).
- Tài liệu đọc trước: DS106 Chapter 3 (pp. 1–15), Chapter 4 *No Free Lunch Theorem* (pp. 1–8), Chapter 5 *Local and Global Optimization* (pp. 1–12); Wolpert & Macready (1997).

**Quy tắc kỹ thuật áp dụng cả tuần**

- **Chỉ qua tracker:** mọi lần đánh giá hàm mục tiêu là `tracker(theta)`. Nga không gọi `make_objective` bên trong thuật toán. `make_objective` chỉ được gọi ở runner khi dựng tracker.
- **Chạy hết ngân sách:** `_search` của cả 5 thuật toán trả về `False`. Dấu hiệu hội tụ sớm ghi vào `extra["diversity"]`, không dùng để dừng.
- **Tái lập từng bit:** RNG do lớp cơ sở cấp, khởi tạo bằng `np.random.default_rng([seed, self.D, zlib.crc32(self.name.encode()), 1])`. Cấm `np.random.seed` toàn cục, cấm `hash()` của Python, cấm tự tạo RNG khác trong thuật toán.
- **DE và PSO tự cài đặt**, cấm `scipy.optimize.differential_evolution`.
- **Single Writer:** chỉ tiến trình chính ghi file kết quả (CSV, JSONL, `.npz`); mỗi dòng CSV `flush()` rồi `os.fsync()`.
- **Cách ly Test:** nhánh Global và Runner không import, không nhắc tới `P_test`, `Y_test`, `*_test.npy`. Runner chỉ nạp dữ liệu Validation qua `data_io.py`.
- **Phạm vi LO/HI:** mọi nghiệm nằm trong `[0.05, 0.95]^D`; tracker là chốt chặn cuối bằng `np.clip`, nhưng thuật toán phải tự xử lý biên.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Quy ước chung cho 5 thuật toán — `src/optimizers/global_/`

- Mỗi lớp có thuộc tính `name` đúng tên đăng ký ở mục 2.8, và đọc siêu tham số từ `hparams` (dict) với giá trị mặc định tuần này ghi trong `configs/hparams_default.yaml`:

  | Thuật toán | Mặc định tuần 1 |
  |---|---|
  | `simulated_annealing` | `T0=5e-3`, `sigma0=0.1`, `T_end=1e-5`, `sigma_end=0.005` |
  | `differential_evolution` | `strategy=rand1`, `F=0.5`, `CR=0.9` |
  | `pso` | `topology=gbest`, `w=0.729`, `c1=c2=1.49445` |
  | `optuna_tpe` | `multivariate=false`, `n_startup_trials=20` |

  Đây là giá trị khởi đầu. Bản chính thức do tuning chọn ở Tuần 2 (LD-7), không quyết định ở tuần này.
- Hàm biên phản xạ dùng chung (`src/optimizers/global_/_common.py`). Chỉ biến đổi các phần tử nằm ngoài miền để giữ nguyên bit của phần tử hợp lệ:

  ```python
  import numpy as np
  from src.contract.constants import LO, HI

  def reflect(x, lo=LO, hi=HI):
      x = np.asarray(x, dtype=np.float64)
      out = (x < lo) | (x > hi)
      if not out.any():
          return x
      w = hi - lo
      y = np.mod(x[out] - lo, 2.0 * w)            # sóng tam giác: phản xạ nhiều lần nếu cần
      x = x.copy()
      x[out] = lo + np.where(y > w, 2.0 * w - y, y)
      return x

  def diversity(pop):
      return float(np.mean(np.std(pop, axis=0)) / (HI - LO))   # std với ddof=0
  ```
- Chữ ký `_search` và các thuộc tính (`tracker.remaining`, `tracker(theta)`, `self.D`, `self.extra`, `x0`) đối chiếu với `base_optimizer.py` của Ngọc. Các đoạn mã dưới đây là đặc tả tham chiếu; thay đổi tên theo hợp đồng thực tế nhưng giữ nguyên thuật toán.

### 3.2. `random_search.py`

- `θ ~ Uniform([0.05, 0.95]^D)` i.i.d., `rng.uniform(LO, HI, size=D)`, lặp đến khi `tracker.remaining == 0`, trả `False`.
- Đây là mốc đối chứng NFLT (Chapter 4): thuật toán nào không vượt được Random Search ở một ô `(D, budget)` thì không khai thác được cấu trúc của hàm tại ô đó.
- Vì khóa RNG không chứa budget, chuỗi mẫu ở NFE=200 là tiền tố của chuỗi ở NFE=5000. Test tận dụng tính chất này (xem 3.10).

### 3.3. `simulated_annealing.py`

Đặc tả:

```python
def _search(self, tracker, x0, rng):
    B = tracker.remaining                      # tổng ngân sách của job (tracker mới tạo)
    cur = np.clip(np.asarray(x0, dtype=np.float64), LO, HI)
    J_cur = tracker(cur)                        # 1 NFE
    for k in range(1, B):
        frac = k / (B - 1)
        T     = self.T0     * (T_END     / self.T0)     ** frac
        sigma = self.sigma0 * (SIGMA_END / self.sigma0) ** frac
        cand = cur.copy()
        if self.D <= 3:
            cand += rng.normal(0.0, sigma, self.D)
        else:                                   # D = 28: mỗi tọa độ nhiễu với xác suất 3/D
            mask = rng.random(self.D) < 3.0 / self.D
            if not mask.any():
                mask[rng.integers(self.D)] = True
            cand[mask] += rng.normal(0.0, sigma, int(mask.sum()))
        cand = np.clip(cand, LO, HI)
        J_new = tracker(cand)                   # 1 NFE
        dJ = J_new - J_cur
        if dJ >= 0 or rng.random() < np.exp(dJ / T):    # Metropolis: P = min(1, exp(ΔJ/T))
            cur, J_cur = cand, J_new
    return False
```

- Khi `ΔJ = 0` (trên plateau) xác suất chấp nhận bằng 1: SA di chuyển tự do trên các vùng phẳng của hàm bậc thang. Ghi chú điều này trong `reports/global_algorithms.md` (Tuần 2).
- Lịch nhiệt phụ thuộc budget `B`: `T_k = T0·(T_end/T0)^(k/(B−1))`, `σ_k = σ0·(σ_end/σ0)^(k/(B−1))`, `T_end = 1e-5`, `σ0 = 0.1`, `σ_end = 0.005`. Căn cứ: bước nhảy tối thiểu của Macro-F1 khoảng `2e-4`, nên `T` cuối nhỏ hơn bước nhảy này để chấp nhận trở nên gần như tham lam.
- Nếu hợp đồng không cấp `x0`, lấy điểm đầu bằng `rng.uniform(LO, HI, D)`.

### 3.4. `differential_evolution.py` (tự cài)

- Quần thể: `NP = max(8, min(10·D, B // 10))`. D=28: NP = 20/100/280 cho B = 200/1000/5000. D=3: 20/30/30. D=1: 10/10/10.
- Khởi tạo đều trong `Θ`, đánh giá toàn bộ (`NP` NFE).
- Đột biến: `rand1`: `v = x_r1 + F·(x_r2 − x_r3)`; `best1`: `v = x_best + F·(x_r1 − x_r2)`. Chỉ số `r*` đôi một khác nhau và khác `i`.
- Lai nhị thức: `mask = rng.random(D) < CR`, ép `mask[j_rand] = True`. Biên: `reflect` (không `clip`).
- Chọn lọc 1–1 với **`≥`**: trial thay target khi bằng hoặc tốt hơn, để quần thể trôi được trên plateau. Ghi quyết định này trong `reports/global_algorithms.md`.
- Cập nhật **đồng bộ theo thế hệ**: đột biến dùng quần thể đầu thế hệ (`x_best` cố định đầu thế hệ), kết quả thay thế ghi vào bản sao `new_pop`. Thế hệ cuối có thể dang dở khi hết ngân sách, và vẫn phải áp dụng các thay thế đã thực hiện.

```python
def _search(self, tracker, x0, rng):
    B = tracker.remaining
    NP = max(8, min(10 * self.D, B // 10))
    pop = rng.uniform(LO, HI, (NP, self.D))
    fit = np.empty(NP)
    for i in range(NP):
        fit[i] = tracker(pop[i])
    self._log_generation(pop, tracker)                  # diversity + gen_end_nfe (NFE đã dùng)
    while tracker.remaining > 0:
        new_pop, new_fit = pop.copy(), fit.copy()
        best = pop[np.argmax(fit)]
        for i in range(NP):
            if tracker.remaining == 0:
                break
            others = np.delete(np.arange(NP), i)
            if self.strategy == "rand1":
                r1, r2, r3 = rng.choice(others, 3, replace=False)
                v = pop[r1] + self.F * (pop[r2] - pop[r3])
            else:                                       # best1
                r1, r2 = rng.choice(others, 2, replace=False)
                v = best + self.F * (pop[r1] - pop[r2])
            mask = rng.random(self.D) < self.CR
            mask[rng.integers(self.D)] = True           # j_rand
            trial = reflect(np.where(mask, v, pop[i]))
            f_trial = tracker(trial)
            if f_trial >= fit[i]:                       # quy tắc ≥
                new_pop[i], new_fit[i] = trial, f_trial
        pop, fit = new_pop, new_fit
        self._log_generation(pop, tracker)
    return False
```

- `extra["diversity"]` (mảng theo thế hệ, gồm cả thế hệ 0 sau khi khởi tạo) và `extra["gen_end_nfe"]` (NFE tích lũy tại cuối mỗi thế hệ; thế hệ dang dở cuối cùng ghi với `gen_end_nfe = B`). Công thức: `diversity = mean_j(std_i(x_ij)) / (HI − LO)`.

### 3.5. `pso.py` (tự cài)

- `NP` như DE. Vận tốc khởi tạo `U(−0.1, 0.1)·(HI − LO)`; kẹp `|v| ≤ 0.2·(HI − LO)`.
- Cập nhật Clerc–Kennedy: `v ← w·v + c1·r1·(pBest − x) + c2·r2·(gBest − x)`, `w = 0.729`, `c1 = c2 = 1.49445`.
- Topology `gbest` (dẫn đường toàn bầy) hoặc `ring` (mỗi hạt dùng `lBest` là `pBest` tốt nhất trong `{i−1, i, i+1}` theo vòng tròn, thay cho `gBest`).
- Cập nhật **đồng bộ theo thế hệ**: đánh giá cả bầy rồi mới cập nhật `pBest`, `gBest`. Vị trí ra ngoài miền được `reflect`, và thành phần vận tốc tương ứng đảo dấu (quyết định thiết kế, ghi trong báo cáo).
- Cập nhật `pBest` bằng `≥` (cùng triết lý plateau với DE).

```python
def _search(self, tracker, x0, rng):
    B = tracker.remaining
    NP = max(8, min(10 * self.D, B // 10))
    span = HI - LO
    pos = rng.uniform(LO, HI, (NP, self.D))
    vel = rng.uniform(-0.1, 0.1, (NP, self.D)) * span
    pfit = np.array([tracker(pos[i]) for i in range(NP)])
    pbest = pos.copy()
    self._log_generation(pos, tracker)
    while tracker.remaining > 0:
        if self.topology == "gbest":
            G = np.broadcast_to(pbest[np.argmax(pfit)], pos.shape)
        else:                                            # ring: lBest trong {i-1, i, i+1}
            nbr = np.stack([(np.arange(NP) - 1) % NP, np.arange(NP), (np.arange(NP) + 1) % NP], axis=1)
            G = pbest[nbr[np.arange(NP), np.argmax(pfit[nbr], axis=1)]]
        r1, r2 = rng.random((NP, self.D)), rng.random((NP, self.D))
        vel = self.w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (G - pos)
        vel = np.clip(vel, -0.2 * span, 0.2 * span)
        new = pos + vel
        out = (new < LO) | (new > HI)
        new = reflect(new)
        vel = np.where(out, -vel, vel)
        pos = new
        for i in range(NP):
            if tracker.remaining == 0:
                break
            f = tracker(pos[i])
            if f >= pfit[i]:
                pfit[i], pbest[i] = f, pos[i]
        self._log_generation(pos, tracker)
    return False
```

- Ghi `extra["diversity"]` và `extra["gen_end_nfe"]` như DE.

### 3.6. `optuna_tpe.py`

```python
def _search(self, tracker, x0, rng):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=self.seed, multivariate=self.multivariate, n_startup_trials=self.n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    def obj(trial):
        theta = np.array([trial.suggest_float(f"t{j}", LO, HI) for j in range(self.D)])
        return tracker(theta)
    study.optimize(obj, n_trials=tracker.remaining, show_progress_bar=False,
                   gc_after_trial=True, n_jobs=1)           # n_jobs=1 để tái lập
    return False
```

- `n_jobs=1` bắt buộc để tái lập. Ghim phiên bản `optuna` trong `requirements`.
- Thời gian sampler không tính vào NFE nhưng tính vào `time_seconds`. Đo thời gian tích lũy trong `obj` (đánh giá tracker) và ghi `extra["tpe_overhead_seconds_per_trial"] = (tổng thời gian − thời gian trong obj)/số trial`.
- Dữ liệu thời gian này phục vụ `reports/global_algorithms.md` và biểu đồ `runtime_vs_nfe.png`.

### 3.7. `registry.py`

- Từ điển ánh xạ **11** mục (10 thuật toán + `constant_0.5`) theo mục 2.8: 5 thuật toán của Nga; `coordinate_ascent`, `nelder_mead`, `powell` (Thy); `adam_surrogate`, `lbfgsb_surrogate` (Vinh). Dùng đăng ký lười (import khi cần) để tuần này chạy được dù thuật toán của Thy và Vinh chưa tích hợp. Tên chính xác theo 2.8.
- `get_optimizer(name, D, seed, hparams)`; thuật toán nhóm gradient nhận `surrogate=make_surrogate(...)` qua tracker (xem 3.8).

### 3.8. `run_benchmark.py` — Batch Runner

**CLI** (union của MD và Word, đã được Ngọc chốt):

```bash
python -m src.runner.run_benchmark \
  --algos all|name1 name2 ... \
  --dims 1 3 28 --budgets 200 1000 5000 --seeds 0-19 \
  --lam <giá trị> | --lam-file artifacts/lambda_star.json \
  --hparams configs/hparams_frozen.yaml \
  --mapping auto|global|freq3|semantic4|per_label \
  --fallback argmax|neutral|head_argmax|none \
  --exclusive-neutral true|false \
  --n-jobs 6 --resume \
  --out outputs/benchmark_results.csv \
  [--use-mock] [--tag <str>] [--config-id <str>]
```

Quy tắc phân giải:

- **`--mapping` và D:** `auto` ánh xạ D=1 → `global`, D=3 → `freq3`, D=28 → `per_label` (D=4 chỉ qua `semantic4` tường minh). Nếu chỉ định `--mapping` mà không có `--dims`, runner tự suy D (`semantic4` → 4, `freq3` → 3, `global` → 1, `per_label` → 28). Nếu `--dims` mâu thuẫn với `--mapping` thì dừng với lỗi rõ ràng.
- **`--lam` / `--lam-file`:** `--lam` (số) ưu tiên hơn file. Tra file theo khóa chuỗi `str(D)`; thiếu khóa (ví dụ D=4) thì **dừng với lỗi**, không điền mặc định ngầm. `D=1` luôn ép `λ=0`.
- **`--tag`:** nếu không có `--out` tường minh thì tên file là `outputs/benchmark_results_<tag>.csv`. `--tag pilot` → `benchmark_results_pilot.csv`; `--tag selective` → `benchmark_results_selective.csv`.
- **`--config-id`:** khi có, thêm cột thứ 9 `config_id` vào CSV (8 cột chuẩn + `config_id`), và là một phần của khóa `--resume`. File chính tuyệt đối không có cột này.
- **Khóa bảo vệ file chính (Ngọc: "Runner từ chối ghi đè file chính"):** `outputs/benchmark_results.csv` và `outputs/benchmark_theta_log.jsonl` chỉ được ghi ở **chế độ chính**: không `--tag`, `--mapping auto`, `--fallback argmax`, `--exclusive-neutral true`, λ lấy từ `--lam-file`, không `--use-mock`, không `--config-id`. Mọi chế độ khác mà `--out` trỏ vào file chính bị từ chối (thoát mã khác 0, không ghi byte nào). `--out` trỏ nơi khác (ví dụ `outputs/sepcheck_results.csv` của Vinh) được phép.
- **File bên cạnh cho lượt không phải chế độ chính:** JSONL ghi vào `<out_stem>_theta_log.jsonl`, quỹ đạo ghi vào `outputs/trajectories_<out_stem>/`, không bao giờ vào `benchmark_theta_log.jsonl` hay `outputs/trajectories/`.

**Kiến trúc thực thi:**

1. **Một khóa = một job:** `(algorithm_name, dimension_D, seed, nfe_budget)` (thêm `config_id` khi có). Dùng `concurrent.futures.ProcessPoolExecutor` với `mp_context=spawn`; dữ liệu Validation nạp một lần mỗi worker qua `initializer`. Đặt `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` (và `OPENBLAS_NUM_THREADS=1`) ở dòng đầu của `src/runner/__init__.py`, trước khi import NumPy.
2. **Worker** nhận khóa, dựng `NFETracker(make_objective(...), max_nfe=budget, surrogate=make_surrogate(...) nếu thuật toán thuộc nhóm gradient)` mới cho mỗi job, chạy `optimize()`, trả về `(row_dict, theta_record, trajectory_dict)` qua giá trị trả về. `λ` và `t_g` lấy theo quy tắc phân giải ở trên và `artifacts/t_g.json`. Worker **không ghi file**.
3. **Tiến trình chính là writer duy nhất.** Thứ tự ghi mỗi job: (a) `.npz` (ghi vào file tạm rồi đổi tên nguyên tử), (b) một dòng JSONL (`flush` + `fsync`), (c) một dòng CSV (`flush` + `fsync`). CSV mở với `newline=""`, `lineterminator="\n"`.
4. **`--resume`:** đọc CSV, tập khóa đã xong; bỏ qua. Trước khi chạy, **đối soát** để chạy lại sau ngắt không sinh trùng: (i) nếu dòng CSV cuối bị cụt (không có `\n`) thì cắt bỏ; (ii) ghi lại JSONL chỉ giữ các dòng có khóa nằm trong tập CSV đã xong (ghi nguyên tử); (iii) `.npz` mồ côi bị ghi đè khi job chạy lại.
5. **`constant_0.5`:** ở chế độ chính, sinh đúng 1 dòng (`dimension_D=1`, `seed=0`, `nfe_budget=1`, `actual_nfe=1`) và 1 bản ghi JSONL, **không sinh `.npz`** (đánh giá tĩnh, được miễn trừ).
6. **Nhật ký** `outputs/run_log.txt`: tiêu đề phiên (git commit hash, phiên bản Python/NumPy/Optuna, SHA-256 của file `--hparams` và `--lam-file`, đầy đủ tham số CLI), mỗi job một dòng (khóa, thời gian, ETA), heartbeat mỗi 60 giây. Lỗi một job ghi stack trace, **không** điền giá trị giả, runner tiếp tục và in danh sách job lỗi ở cuối (thoát mã khác 0 nếu có job lỗi).
7. **Chống treo:** bắt `BrokenProcessPool` (worker chết do hết bộ nhớ), ghi nhật ký, dựng lại pool và nộp lại các job chưa xong (tối đa 3 lần), sau đó dừng và báo danh sách job lỗi.
8. Thứ tự nộp job: giảm dần theo chi phí ước tính (ví dụ `optuna_tpe`, D=28, B=5000 trước) để rút ngắn tổng thời gian; thứ tự dòng trong CSV không mang ý nghĩa.

### 3.9. `validate_results.py`

```bash
python -m src.runner.validate_results --results outputs/benchmark_results.csv --mode full
python -m src.runner.validate_results --results outputs/benchmark_results_pilot.csv --mode custom --expected-rows 20
```

Kiểm tra (chế độ `full`):

1. Header đúng 8 cột theo thứ tự 2.9 (thêm `config_id` ở cuối nếu là file có thẻ).
2. Không trùng khóa; kiểu dữ liệu đúng; không NaN.
3. `1 ≤ actual_nfe ≤ nfe_budget`; `is_converged=False ⇒ actual_nfe == nfe_budget`.
4. Số dòng `== 1801` (chế độ `custom`: bằng `--expected-rows`).
5. Mỗi khóa có bản ghi trong JSONL tương ứng (1.801) và file `.npz` tương ứng (1.800): **miễn trừ rõ ràng** cho dòng `algorithm_name == "constant_0.5"`.
6. In `PASS` hoặc danh sách vi phạm cụ thể, thoát mã khác 0 nếu vi phạm.

### 3.10. Kiểm thử trên mock — `tests/test_global_optimizers.py`, `tests/test_runner.py`

`tests/test_global_optimizers.py` (dữ liệu `mock_fullshape`):

- **Dừng đúng ngân sách:** 5 thuật toán × budget {200, 1000, 5000} × D {1, 3, 28}: bộ đếm độc lập (bọc hàm mục tiêu bằng closure đếm) bằng `tracker.nfe` bằng `actual_nfe` bằng budget, `is_converged=False`.
- **Biên:** mọi nghiệm do DE/PSO sinh ra nằm trong `[LO, HI]` ngay cả khi không qua tracker; `reflect` đúng với giá trị vượt biên nhiều lần (ví dụ 2.7, −1.3).
- **RNG:** thuật toán dùng RNG do lớp cơ sở cấp và khóa khớp `[seed, D, crc32(name), 1]`; không tồn tại `np.random.seed` hay `hash(` trong mã nguồn (kiểm bằng grep trong test).
- **Random Search:** với cùng seed, `best_score` ở budget lớn hơn **không nhỏ hơn** budget nhỏ hơn (tiền tố RNG); trung bình theo seeds không giảm theo budget.
- **Diversity:** DE/PSO có `extra["diversity"]` và `extra["gen_end_nfe"]` cùng độ dài, `gen_end_nfe` tăng dần, kết thúc bằng budget.
- **SA:** trên hàm hằng, mọi bước được chấp nhận (`ΔJ = 0` → xác suất 1).
- **TPE:** hai lần chạy cùng seed cho `best_score` giống từng bit.

`tests/test_runner.py`:

- **Tái lập:** chạy lại 10 khóa ngẫu nhiên, `best_val_score` giống từng bit. Chạy `--n-jobs 1` và `--n-jobs 4` cho cùng tập khóa và cùng giá trị (so sánh sau khi sắp theo khóa, bỏ cột `time_seconds`).
- **Resume:** chạy tập con, dùng `SIGKILL` giữa chừng, chạy lại với `--resume`: không có khóa trùng, kết quả cuối bằng lần chạy tham chiếu; thử cả kịch bản dòng CSV cuối bị cụt.
- **Single Writer:** chỉ PID của tiến trình chính mở file kết quả (kiểm bằng ghi nhật ký PID ở lớp writer).
- **`constant_0.5`:** đúng 1 dòng, 1 bản ghi JSONL, không `.npz`.
- **Khóa bảo vệ:** `--tag selective --out outputs/benchmark_results.csv` bị từ chối và file chính không đổi; mọi cờ khác chế độ chính mà `--out` là file chính đều bị từ chối.
- **Phân giải D/λ:** `--mapping semantic4` tự suy D=4; `--dims 3 --mapping semantic4` báo lỗi; thiếu khóa λ cho D=4 báo lỗi; `D=1` ép `λ=0` kể cả khi `--lam 0.1`.
- **`validate_results`:** các ca âm tính (header sai, khóa trùng, thiếu `.npz`, thiếu dòng JSONL, `is_converged=False` nhưng `actual_nfe < budget`) đều bị phát hiện; ca miễn trừ `constant_0.5` đạt.
- **Smoke test `--use-mock`** toàn lưới thu gọn (ví dụ 2 seeds, budget 200), đủ cho 5 thuật toán của Nga.

### 3.11. Pilot báo cáo sơ bộ (Ngày 4–5)

Sau khi Thy bàn giao `P_val`, `Y_val`, `t_g.json` thật:

```bash
python -m src.runner.run_benchmark \
  --algos random_search optuna_tpe --dims 3 --budgets 200 1000 --seeds 0-4 \
  --lam 0.1 --hparams configs/hparams_default.yaml \
  --mapping auto --fallback argmax --exclusive-neutral true \
  --n-jobs 4 --resume --tag pilot
```

- Ra `outputs/benchmark_results_pilot.csv` (20 dòng: 2 thuật toán × 2 budget × 5 seeds), không bao giờ vào file chính.
- `λ = 0.1` là giá trị quy ước tạm thời, thống nhất với quy ước tuning của Tuần 2. Phối hợp với Thy để pilot Coordinate Ascent dùng cùng λ, nhằm Ngọc ghép được vào một bảng so sánh.
- Tạo `outputs/tables/pilot_summary_week1.csv`: `algorithm, nfe_budget, mean, std, time_seconds_mean` và bàn giao cho Ngọc. Ghi rõ trong tệp và khi bàn giao rằng đây là số liệu pilot với cấu hình mặc định, không dùng cho kết luận khoa học.
- Chạy `validate_results --mode custom --expected-rows 20` trên file pilot.

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Thuật toán | `src/optimizers/global_/random_search.py`, `simulated_annealing.py`, `differential_evolution.py`, `pso.py`, `optuna_tpe.py`, `_common.py` |
| Runner | `src/runner/registry.py`, `run_benchmark.py`, `validate_results.py`, `src/runner/__init__.py` |
| Cấu hình | `configs/hparams_default.yaml` |
| Test | `tests/test_global_optimizers.py`, `tests/test_runner.py` |
| Dữ liệu pilot | `outputs/benchmark_results_pilot.csv`, `outputs/benchmark_results_pilot_theta_log.jsonl`, `outputs/trajectories_benchmark_results_pilot/*.npz`, `outputs/run_log.txt` |
| Bảng | `outputs/tables/pilot_summary_week1.csv` (bàn giao Ngọc) |

---

## 5. Definition of Done (DoD tuần)

- [ ] Cả 5 thuật toán kế thừa `BaseOptimizer`, chỉ cài `_search`, không ghi đè `optimize()`, mọi đánh giá qua `tracker`, không gọi `make_objective`.
- [ ] 5 thuật toán × 3 budget × 3 D dừng đúng `max_nfe` trên `mock_fullshape` (bộ đếm độc lập), `_search` trả `False`.
- [ ] DE và PSO tự cài, không dùng `scipy.optimize.differential_evolution`, biên phản xạ, ghi `diversity` và `gen_end_nfe`.
- [ ] Runner chạy ổn định với `--n-jobs 4`, `--resume` không sinh dòng trùng (kể cả sau `SIGKILL`), Single Writer có `flush` + `fsync`, `constant_0.5` đúng quy ước.
- [ ] Khóa bảo vệ file chính hoạt động, cờ `--lam`, `--mapping`, `--fallback`, `--exclusive-neutral`, `--tag`, `--config-id`, `--use-mock` đều hoạt động, D tự suy từ `--mapping`.
- [ ] `validate_results.py` bắt đủ các ca âm tính và miễn trừ `.npz` cho `constant_0.5`.
- [ ] `pytest tests/test_global_optimizers.py tests/test_runner.py` đạt 100% PASS, gồm kiểm tra tái lập 10 khóa và `--n-jobs 1` so với `--n-jobs 4`.
- [ ] Pilot Random Search và Optuna TPE (D=3, NFE 200 và 1000, 5 seeds) trên dữ liệu thật đã bàn giao Ngọc kèm ghi chú λ quy ước.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py` (không có chuỗi cấm trong `src/optimizers/global_` và `src/runner`).
- [ ] Ngọc duyệt chuẩn `BaseOptimizer` trong PR.