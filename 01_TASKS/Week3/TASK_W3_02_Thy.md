# TASK_02_Thy_Week3 — Giám sát runtime benchmark và khung Bảng 4

| Mục | Nội dung |
|---|---|
| Vai trò | Nhánh Core: xác nhận và báo cáo nhóm thuật toán cục bộ, khởi tạo Bảng 4 |
| Owner | Thy (Bảng 4, sanity check, báo cáo xác nhận); Nga chịu trách nhiệm điều phối batch runner |
| Reviewers | Ngọc (Leader, nghiệm thu), Vinh (Peer) |
| Phối hợp | Nga (batch runner), Vinh (`lambda_star.json`) |
| Mã task gốc | CORE-01 |
| Khung thời gian | Tuần 3 trong kế hoạch 4 tuần, mỗi tuần 7 ngày lịch |
| Deadline tuần | 23:30 ngày cuối Tuần 3 |

## 1. Mục tiêu tuần (Weekly Goal)

1. Xác nhận và báo cáo việc 3 thuật toán cục bộ chạy trơn tru trong mẻ benchmark lớn (1.801 lượt) của Nga.
2. Dựng khung Bảng 4 cho 28 nhãn từ dữ liệu train và Val.
3. Kiểm tra tính nhất quán toán học giữa vector nghiệm trong log và điểm số trong CSV kết quả.

**Cách ly Test:** trong Tuần 3 Thy tuyệt đối không đụng vào tập Test. Các cột Test của Bảng 4 để trống đến Tuần 4.

## 2. Input đầu vào

- Nhánh `main` đã tích hợp đủ 10 thuật toán và cấu hình siêu tham số đóng băng.
- `artifacts/lambda_star.json` (λ* do Vinh xác định bằng 5-fold CV).
- `outputs/run_log.txt`, `benchmark_theta_log.jsonl`, `benchmark_results.csv` (do runner của Nga xuất ra).
- `P_val.npy`, `Y_val.npy`, `train_config.json` (chứa `n_c`, `N_train`), `mapping_freq3.json`, `reports/baseline_reproduction.md`.
- Tài liệu bắt buộc đọc:
  - Slide DS106 Chapter 3 *Optimization Algorithm Selection* (pp. 1–15).
  - Bartz-Beielstein et al. (2020), *Benchmarking in Optimization: Best Practice and Open Issues*.
  - Mục tiêu: nắm quy tắc đo runtime và ngân sách NFE cố định khi chạy song song đa tiến trình.

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Xác nhận và báo cáo runtime của 3 thuật toán cục bộ

Thy không sở hữu runner. Vai trò của Thy là theo dõi, phân tích và báo lỗi cho Nga. Thy không sửa runner khi chưa thống nhất với Nga. Thy đọc `outputs/run_log.txt` theo định dạng thực tế Nga công bố, không giả định tên cột.

Quy mô tham chiếu (suy ra từ cấu hình kế hoạch, cần Nga xác nhận): 3 thuật toán cục bộ × 20 seeds × 3 D × 3 budgets = 540 lượt. Con số 1.801 phù hợp với 10 thuật toán × 180 lượt + 1 dòng baseline.

Với 3 thuật toán cục bộ, Thy kiểm tra và ghi lại:
- Số lượt hoàn thành, số lượt lỗi hoặc timeout.
- `actual_nfe ≤ nfe_budget` ở mọi dòng.
- Thời gian trung vị mỗi lượt theo (thuật toán, D, budget).
- Tỷ lệ Nelder-Mead hội tụ sớm (`is_converged=True` và NFE đã dùng nhỏ hơn budget).
- Tỷ lệ Coordinate Ascent hoàn thành ít nhất một lượt quét ở budget 200.
- Dấu hiệu tranh chấp tài nguyên: worker chậm bất thường, tràn bộ nhớ. Xác nhận `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` đều bằng 1 để không làm lệch phép đo 3 ms.
- Nếu lỗi phát sinh từ mã của Thy: tái hiện trên `mock_fullshape` với cùng seed, sửa trong module của Thy và gửi PR. Nếu lỗi từ runner: báo Nga kèm log và lệnh tái hiện.

Kết quả gom vào `reports/runtime_monitoring_local.md`. Mọi số liệu dẫn từ log, không gõ tay.

### 3.2 Khung Bảng 4 (`outputs/tables/per_label_table4.csv`)

Thy viết `scripts/build_table4_frame.py`, **không** đặt trong `src/optimizers/` hay `landscape.py`, và chỉ nạp Val qua `data_io`. Nguồn `n_c` và `N_train` là `train_config.json`.

| Cột | Nội dung | Trạng thái Tuần 3 |
|---|---|---|
| `idx` | Chỉ số cột lớp 0 đến 27 | Điền |
| `label_en` | Tên nhãn tiếng Anh | Điền |
| `n_train` | Số mẫu dương trong train (`n_c`) | Điền |
| `freq_pct` | `100 · n_c / N_train` | Điền |
| `group_freq3` | Head, Mid hoặc Tail theo `mapping_freq3.json` | Điền |
| `is_neutral` | Cờ Neutral | Điền |
| `f1_base_val` | F1 từng nhãn tại ngưỡng 0.5 trên Val | Điền |
| `f1_base_test` | F1 từng nhãn tại ngưỡng 0.5 trên Test | Để trống, điền ở Tuần 4 |
| `t_star`, `f1_opt_val`, `f1_opt_test`, `delta_f1_val`, `delta_f1_test` | Kết quả sau tối ưu | Để trống, điền ở Tuần 4 |

- `f1_base_val` tính tại **ngưỡng cố định 0.5**, qua đúng đường `predict_labels` (mặc định `exclusive_neutral=True`, fallback `argmax`), không tuning.
- Kiểm tra chéo (chỉ Val):
  - `f1_base_val` khớp `sklearn.metrics.f1_score(average=None, zero_division=0)` ở `1e-9`.
  - Trung bình `f1_base_val` bằng Val Macro-F1 baseline trong `reports/baseline_reproduction.md`, sai số ≤ `1e-9`.
  - `n_train` khớp `train_config.json` và `meta.json`.
- F1@0.5 trên Test: Thy không tính trong Tuần 3. Nguồn là mục Test Macro-F1 trong báo cáo Tuần 1 (cấp độ Macro) và cột `f1_base_test` do `src/eval/test_gate.py` điền ở Tuần 4 (cấp độ từng nhãn).

### 3.3 Sanity check tính nhất quán toán học (chỉ Val)

1. Lấy mẫu ngẫu nhiên các dòng từ `benchmark_theta_log.jsonl`, phủ đều các thuật toán, D và λ (ít nhất 30 dòng, ưu tiên cả 3 thuật toán cục bộ).
2. Nạp lại `theta`, dựng `make_objective` bằng đúng `mapping_mode`, `fallback_strategy`, `exclusive_neutral`, `t_g` của lượt chạy, và đúng λ ghi trong dòng log. Với các lượt dùng λ*, nạp λ* từ `artifacts/lambda_star.json` và xác nhận bằng giá trị trong log.
3. So `J_λ` tính lại với `best_val_score` trong `benchmark_results.csv`. Phải so `J_λ` chứ không so `macro_f1` trần, vì khi λ > 0 điểm số đã trừ phần phạt.
4. Yêu cầu khớp từng bit. Nếu `benchmark_theta_log.jsonl` lưu `theta` đủ độ chính xác (`repr` của float) thì kết quả phải bằng nhau tuyệt đối. Nếu log làm tròn `theta`, ghi rõ và dùng ngưỡng `1e-9`.
5. Mọi dòng sai lệch được ghi vào báo cáo và báo Nga, Ngọc để tìm nguyên nhân (làm tròn log, lệch phiên bản mã, khác λ).

### 3.4 Chuẩn bị cho Tuần 4
- Chạy thử đường xử lý Bảng 4 trên `mock_fullshape` để phát hiện lỗi sớm. Không dùng Test thật.
- Chuẩn bị đoạn mã điền các cột còn trống ngay khi nhận θ*.
- Đọc trước Pillai et al. (2013) và Demszky et al. (2020).

## 4. Output tuần

- `outputs/tables/per_label_table4.csv` (đã điền thống kê train và F1 baseline Val; các cột Test và sau tối ưu để trống).
- `scripts/build_table4_frame.py`.
- `reports/runtime_monitoring_local.md` (báo cáo xác nhận runtime nhóm cục bộ, kèm kết quả sanity check và danh sách dòng đã kiểm tra).

## 5. Definition of Done (DoD Tuần 3)

- [ ] Thy đã đối soát log và xác nhận bằng báo cáo rằng 3 thuật toán cục bộ chạy trơn tru trong mẻ 1.801 lượt của Nga: số lượt hoàn thành, không lỗi, `actual_nfe ≤ nfe_budget` ở mọi dòng. Nếu có lỗi thì báo cáo nêu nguyên nhân và người xử lý.
- [ ] Khung Bảng 4 chính xác về `n_train`, `freq_pct`, `group_freq3`, `is_neutral` và `f1_base_val` cho cả 28 nhãn; khớp sklearn ở `1e-9`.
- [ ] Sanity check bằng `make_objective` cùng λ của từng lượt (λ* với các lượt dùng λ*) được Ngọc nghiệm thu.
- [ ] Thy không nạp tập Test ở bất kỳ mã nào trong tuần.
- [ ] `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` không trả về kết quả nào.