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
Ủng hộ nếu, tại NFE=5000 và cả hai D ∈ {3, 28}: (a) trung vị `nfe_to_99pct` của `adam_surrogate` hoặc `lbfgsb_surrogate` nhỏ hơn trung vị của mọi thuật toán hộp đen; (b) trung vị `best_val_score` của thuật toán gradient đó thấp hơn thuật toán hộp đen tốt nhất với Holm-corrected p < 0.05.

## NFLT — No Free Lunch
Ủng hộ nếu đồng thời: (a) có ít nhất 2 thuật toán khác nhau giữ hạng nhất trong 9 ô (D, nfe_budget); (b) `random_search` không đứng cuối ở ít nhất một ô.

## Phân tích khám phá (không dùng để kết luận H1–H4)
Phân tích hội tụ sớm (Chapter 6), chi phí thời gian thật của TPE, so sánh `freq3` với `semantic4`.