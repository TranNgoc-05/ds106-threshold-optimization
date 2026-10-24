# CHANGE REQUESTS (CR Log) — contract v1.0

| Mã | Nội dung | Tác động | Trạng thái |
|---|---|---|---|
| CR-001 | OI-1 chốt phương án (a): giữ `J_λ` theo LD-8. Phạt L2 là regularization kéo nghiệm về `t_g`; coupling đến hoàn toàn từ Neutral độc quyền và Fallback | Báo cáo, README, `ablation` | Ngọc đã ký |
| CR-002 | Bỏ hoàn toàn CafeBERT; chỉ dùng ViSoBERT | Schema `02_SAMPLE_DATA`, `EVAL-01`, Bảng 2, rủi ro | Ngọc đã ký |
| CR-003 | Quy tắc Neutral độc quyền theo so sánh xác suất chặt | `decision.py`, golden cases, differential test, ablation | Ngọc đã ký |
| CR-004 | Đồng bộ phân vai: Thy = Core/Local, Nga = Global/Runner, Vinh = Surrogate/Eval | Cập nhật cấu trúc task `01_TASKS/`, ma trận cross-review | Ngọc đã ký |
| CR-005 | Bổ sung định dạng tối thiểu cho log: `tuning_log.csv` thêm cột `nfe_budget`, `split`; `cv_lambda_results.csv` có cột `dimension_D, lam, mean_heldout_macro_f1, se`; `registry.py` xuất biến `REGISTRY: Dict[str, Type[BaseOptimizer]]`; `selective_pressure_study.csv` có cột `algorithm, variant, seed, best_val_score, diversity_collapse_nfe`; `stats_mwu_holm.csv` dùng giá trị `primary` / `exploratory` cho cột `family` | Phục vụ script kiểm toán G2, G3, G5 tự động | Ngọc đã ký |