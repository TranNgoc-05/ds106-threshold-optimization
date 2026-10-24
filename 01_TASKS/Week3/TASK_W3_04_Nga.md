# TASK_04_Nga_Week3 — Benchmark chính 1.801 lượt, Cổng G4 & nghiên cứu áp lực chọn lọc

| Mục | Nội dung |
|---|---|
| **Vai trò** | Nhánh Global: quản lý mẻ benchmark lớn nhất của dự án và nghiên cứu hội tụ sớm (Chapter 6) |
| **Owner** | Nga |
| **Reviewers** | Ngọc (Leader, A: nghiệm thu G4, cấp `LOCK_FREEZE.json`), Vinh (Peer Reviewer: tái lập, thống kê) |
| **Deadline** | Cuối Tuần 3, tương ứng mốc M4 (benchmark 1.801 dòng) và nghiên cứu áp lực chọn lọc |
| **Phối hợp** | Vinh (C: cung cấp `lambda_star.json`, chạy `sep_check`), Thy (C), Ngọc (A) |

---

## 1. Mục tiêu tuần (Weekly Goal)

1. Chạy **một lần duy nhất** mẻ benchmark chính 1.801 lượt trên tập Validation, đảm bảo không treo hoặc chết âm thầm.
2. Kiểm tra toàn vẹn bằng `validate_results.py` và bàn giao cho Ngọc nghiệm thu Cổng G4, tạo `LOCK_FREEZE.json`.
3. Thực hiện nghiên cứu áp lực chọn lọc và hội tụ sớm: 8 biến thể DE/PSO ở D=28, NFE=5000, 20 seeds.
4. Điều phối CPU để hỗ trợ Vinh chạy `sep_check` mà không làm chậm mẻ chính.

---

## 2. Input đầu vào

- Nhánh `main` đã tích hợp đủ 10 thuật toán.
- `configs/hparams_frozen.yaml` (Ngọc đã ký ở Tuần 2).
- `artifacts/lambda_star.json` do Vinh cấp (`{"3": x, "28": y}`), `artifacts/t_g.json` của Thy.
- Tài liệu đọc trước: DS106 Chapter 6 *Premature Convergence* (pp. 1–9); Bartz-Beielstein et al. (2020).

**Quy tắc kỹ thuật áp dụng cả tuần**

- **Chạy chính một lần duy nhất** sau G3; **không chỉnh tay** file kết quả, không mở và lưu lại CSV bằng bảng tính (tránh đổi ký tự cuối dòng làm sai SHA-256).
- Ngọc chỉ băm hai file: `outputs/benchmark_results.csv` (1.801 dòng) và `outputs/benchmark_theta_log.jsonl`. Mọi lượt chạy khác ghi ra file riêng.
- Nghiên cứu áp lực chọn lọc **không được** dùng để đổi cấu hình của benchmark chính (tránh chọn sau khi đã thấy kết quả).
- Cách ly Test: không import hay nhắc `P_test`, `Y_test`, `_test.npy`.

---

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1. Kiểm tra trước khi chạy (preflight)

1. Xác nhận nhánh sạch, ghi commit hash; `configs/hparams_frozen.yaml` đúng bản Ngọc đã ký; `lambda_star.json` có khóa `"3"` và `"28"`.
2. Chạy thử toàn lưới thu gọn bằng `--use-mock --tag preflight` (2 seeds, budget 200) cho đủ 10 thuật toán, rồi `validate_results --mode custom`.
3. Cố định biến môi trường `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; kiểm tra dung lượng đĩa và số lõi; chạy trong `tmux` hoặc `nohup`.
4. Đảm bảo `outputs/benchmark_results.csv` chưa tồn tại, hoặc nếu tồn tại thì chỉ do lần chạy chính bị ngắt (`--resume` sẽ tiếp tục).

### 3.2. Khởi chạy mẻ benchmark chính

```bash
python -m src.runner.run_benchmark \
  --algos all \
  --dims 1 3 28 --budgets 200 1000 5000 --seeds 0-19 \
  --lam-file artifacts/lambda_star.json \
  --hparams configs/hparams_frozen.yaml \
  --mapping auto --fallback argmax --exclusive-neutral true \
  --n-jobs 6 --resume \
  --out outputs/benchmark_results.csv
```

- Lưới: 10 thuật toán × 3 D × 3 budget × 20 seeds = 1.800 lượt + `constant_0.5` = **1.801** dòng. `D=1` ép `λ=0`; `D=3`, `D=28` lấy λ từ `lambda_star.json`.
- Mẻ chính là **ưu tiên số 1** và chạy **một mình** (không chạy song song bất kỳ mẻ nào khác), vì `time_seconds` của file chính phải đo trong điều kiện không tranh chấp CPU giữa các mẻ.
- Chống treo: theo dõi `outputs/run_log.txt` (`tail -f`), heartbeat mỗi 60 giây, ETA; nếu không có job nào hoàn tất quá 3 lần thời gian ước lượng của job chậm nhất (theo `profiling_runtime.csv`) thì điều tra tiến trình. Runner tự xử lý `BrokenProcessPool` (dựng lại pool, nộp lại job chưa xong). Khi tiến trình bị ngắt, chạy lại đúng lệnh trên (`--resume` không sinh dòng trùng).
- Job lỗi: runner ghi stack trace, không điền giá trị giả, in danh sách ở cuối. Sửa nguyên nhân rồi chạy lại bằng `--resume` cho tới khi đủ 1.801 dòng. Nếu phát hiện lỗi thuật toán sau khi đã có kết quả thì **dừng** và báo Ngọc, không sửa tay file kết quả.
- Ghi nhận: `outputs/benchmark_results.csv`, `outputs/benchmark_theta_log.jsonl`, `outputs/trajectories/*.npz` (best_so_far, loại lượt gọi, độ đa dạng quần thể), `outputs/run_log.txt`.

### 3.3. Kiểm tra toàn vẹn và nghiệm thu Cổng G4

```bash
python -m src.runner.validate_results --results outputs/benchmark_results.csv --mode full \
  | tee 03_SUBMISSIONS/Nga/test_logs/validate_results_output.txt
```

Phải đạt `PASS`:

- Đúng **1.801 dòng**, đúng 8 cột chuẩn theo thứ tự, không trùng khóa chính.
- `actual_nfe == nfe_budget` với mọi dòng `is_converged = False`; `1 ≤ actual_nfe ≤ nfe_budget`.
- **1.801 bản ghi** trong `benchmark_theta_log.jsonl` và **1.800 file `.npz`** (miễn trừ rõ ràng cho `constant_0.5`).

Bàn giao cho Ngọc: hai file chính **nguyên trạng**, không sửa tay, và log `validate_results`. Ngọc tính SHA-256 trên môi trường Linux/CI và ban hành `src/artifacts/LOCK_FREEZE.json` (Cổng G4). Từ thời điểm này, không chạy lại hay ghi thêm vào hai file đã băm. Nếu có sai lệch băm (ví dụ do CRLF/LF), đề nghị Ngọc tính lại trên Linux/CI, không sửa tay file.

### 3.4. Nghiên cứu áp lực chọn lọc và hội tụ sớm (Chapter 6)

Mục tiêu: định lượng vai trò của áp lực chọn lọc trong hiện tượng hội tụ sớm của DE và PSO. Đây là thí nghiệm riêng, **không** thay đổi cấu hình benchmark chính.

**Thiết kế 8 biến thể (D=28, NFE=5000, seeds 0–19, λ=λ\*₂₈, các tham số khác theo YAML đã đóng băng):**

| Thuật toán | Biến thể (`config_id`) | Mức áp lực |
|---|---|---|
| DE | `de_rand1_F0.5`, `de_rand1_F0.8`, `de_best1_F0.5`, `de_best1_F0.8` (CR = 0.9) | `best1` cao hơn, F nhỏ cao hơn. Cao nhất `best1 + 0.5`, thấp nhất `rand1 + 0.8` |
| PSO | `pso_gbest_w0.4`, `pso_gbest_w0.729`, `pso_ring_w0.4`, `pso_ring_w0.729` | `gbest` và `w` nhỏ cao hơn. Cao nhất `gbest + 0.4`, thấp nhất `ring + 0.729` |

- Mỗi biến thể có YAML riêng `configs/selective_pressure/<config_id>.yaml`. Chạy 8 lệnh, mỗi biến thể một lệnh, cùng ghi vào một file:

  ```bash
  python -m src.runner.run_benchmark --algos differential_evolution --dims 28 --budgets 5000 --seeds 0-19 \
    --lam-file artifacts/lambda_star.json --hparams configs/selective_pressure/de_best1_F0.5.yaml \
    --config-id de_best1_F0.5 --mapping auto --fallback argmax --exclusive-neutral true \
    --n-jobs 6 --resume --tag selective
  ```
- `--tag selective` ghi dữ liệu thô ra `outputs/benchmark_results_selective.csv` (8 cột chuẩn + `config_id`), JSONL ra `outputs/benchmark_results_selective_theta_log.jsonl`, quỹ đạo ra `outputs/trajectories_benchmark_results_selective/`. Runner **từ chối** ghi vào `benchmark_results.csv`. Tổng: 8 × 20 = **160** dòng.
- **Ưu tiên số 2:** chạy sau khi mẻ chính hoàn tất (xem 3.5).

**Chỉ số cho mỗi biến thể** (script `src/analysis/selective_pressure.py`, đọc `.npz`):

- `best_val_score` (mean ± std qua 20 seeds).
- Đường cong `diversity` theo NFE: trung vị và dải phân vị 25–75 qua 20 seeds (nội suy bậc thang lên lưới NFE chung).
- `diversity_collapse_nfe`: giá trị `gen_end_nfe` đầu tiên có `diversity < 0.01`; `None` nếu không xảy ra. Báo cáo trung vị trên các seed có collapse và số seed không collapse.
- Khoảng cách tới nghiệm tốt nhất mọi thời đại của cùng seed: `gap = max(best_val_score của 8 biến thể cùng seed) − best_val_score(biến thể)`. Thêm cột phụ `gap_to_cell_best` so với giá trị lớn nhất của ô `(D=28, NFE=5000)` trong file chính (đọc, không ghi).

**Xuất:**

- `outputs/tables/selective_pressure_study.csv` (một dòng mỗi biến thể: `algorithm, config_id, pressure_level, best_val_score_mean, best_val_score_std, diversity_collapse_nfe_median, n_seeds_without_collapse, gap_mean, gap_to_cell_best_mean`).
- `outputs/figures/premature_convergence_de_pso.png`: hai panel (DE, PSO), mỗi panel 4 đường trung vị `diversity` theo NFE với dải 25–75, vạch ngưỡng 0.01, thang trục y logarit.
- `reports/selective_pressure_study.md`: báo cáo chuyên sâu, kết luận **có số** (ví dụ `diversity_collapse_nfe` trung vị của `best1` so với `rand1`), **kể cả khi kết quả trái dự đoán**. So sánh suy diễn (nếu cần) do Vinh cấp, Nga chỉ báo cáo mô tả. Với PSO ghi chú rằng `w = 0.4` cùng `c1 = c2 = 1.49445` nằm ngoài tổ hợp co Clerc–Kennedy chuẩn.

### 3.5. Điều phối CPU và hỗ trợ Vinh (thứ tự ưu tiên đã chốt)

| Ưu tiên | Khối lượng | Ghi chú |
|---|---|---|
| 1 (Tuần 3) | Mẻ chính 1.801 lượt | Chạy một mình để `time_seconds` không bị nhiễu; Ngọc cần để ký G4 |
| 2 (Tuần 3) | 160 lượt áp lực chọn lọc (`--tag selective`) và 600 lượt `sep_check` của Vinh | Sau khi mẻ chính xong; phân chia lõi giữa hai mẻ (mỗi mẻ không vượt một nửa số lõi) để tránh vượt ngưỡng 12 giờ máy; `sep_check` tắt Neutral độc quyền và Fallback nên mỗi lượt nhanh hơn >15% |
| 3 (đầu Tuần 4) | ~400 lượt bổ sung (D=4, D=28 λ=0) và ~160 lượt ablation của Vinh | Ghi ra file riêng; phải **hoàn tất trước khi Vinh gọi `load_test()` lần đầu** |

- Nga cấp quyền và hỗ trợ Vinh chạy runner cho `sep_check`: `--mapping per_label --fallback none --exclusive-neutral false --lam <λ*₂₈> --out outputs/sepcheck_results.csv` (đúng 10 thuật toán × budget {200, 1000, 5000} × 20 seeds).
- Cho cấu hình bổ sung: `--mapping semantic4` (D tự suy = 4, cần `--lam <λ*₃>` tường minh vì `lambda_star.json` không có khóa D=4); `--mapping per_label --lam 0`.
- Không chạy bất kỳ lượt nào ghi vào hai file đã băm sau khi `LOCK_FREEZE.json` ra đời.

---

## 4. Output tuần

| Loại | Đường dẫn |
|---|---|
| Kết quả chính | `outputs/benchmark_results.csv` (1.801 dòng), `outputs/benchmark_theta_log.jsonl` (1.801 bản ghi), `outputs/trajectories/*.npz` (1.800 file) |
| Nhật ký | `outputs/run_log.txt`, `03_SUBMISSIONS/Nga/test_logs/validate_results_output.txt` |
| Nghiên cứu áp lực | `outputs/benchmark_results_selective.csv` (160 dòng), `outputs/tables/selective_pressure_study.csv`, `outputs/figures/premature_convergence_de_pso.png`, `reports/selective_pressure_study.md` |
| Mã nguồn | `src/analysis/selective_pressure.py`, `configs/selective_pressure/*.yaml` |
| Bàn giao | Hai file chính nguyên trạng cho Ngọc (băm SHA-256, tạo `LOCK_FREEZE.json`) |

---

## 5. Definition of Done (DoD tuần)

- [ ] `validate_results.py` trả `PASS` trên `benchmark_results.csv`: đúng 1.801 dòng, 8 cột đúng thứ tự, không trùng khóa, `actual_nfe == nfe_budget` khi `is_converged=False`.
- [ ] Đủ 1.801 bản ghi trong `benchmark_theta_log.jsonl` và 1.800 file `.npz` (miễn trừ `constant_0.5`).
- [ ] Mẻ chính chạy đúng một lần, không có file nào bị chỉnh tay, không có job lỗi tồn đọng, nhật ký ghi commit hash, SHA-256 của YAML và `lambda_star.json`.
- [ ] Ngọc nghiệm thu Cổng G4 và ban hành `LOCK_FREEZE.json`.
- [ ] `selective_pressure_study.csv` đủ 8 biến thể × 20 seeds (160 dòng thô) kèm `diversity_collapse_nfe` cụ thể; báo cáo nêu kết luận có số kể cả khi trái dự đoán; biểu đồ `premature_convergence_de_pso.png` đã xuất.
- [ ] Dữ liệu áp lực chọn lọc nằm ở `outputs/benchmark_results_selective.csv`, không có dòng nào lọt vào `benchmark_results.csv`.
- [ ] Vinh đã nhận đủ hỗ trợ runner cho `sep_check`; lượt bổ sung/ablation có kế hoạch hoàn tất trước khi Vinh gọi `load_test()`.
- [ ] Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` chỉ trả về `src/eval/test_gate.py`.