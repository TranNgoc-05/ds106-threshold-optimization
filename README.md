# Tối ưu hóa Vector Ngưỡng Quyết định Đa nhãn trên ViGoEmotions: Khảo sát Địa hình Hàm số, So sánh Hiệu năng Thuật toán và Kiểm chứng Định lý No Free Lunch

Đồ án môn học **DS106 – Tối ưu hóa ứng dụng**, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM.  
**Giảng viên hướng dẫn:** TS. Nguyễn-Tất B. Thiện.  
**Nhóm sinh viên thực hiện:** 
- **Ngọc:** Technical Project Manager & System Architect (Quản trị hợp đồng, CI, nghiệm thu chất lượng G1–G5, báo cáo).
- **Thy:** Core & Local Optimization (Huấn luyện ViSoBERT, hàm ra quyết định, hàm mục tiêu, khảo sát địa hình, Coordinate Ascent, Nelder-Mead, Powell, Bảng 4).
- **Nga:** Global Optimization & Batch Runner (Random Search, SA, DE, PSO, TPE, hệ thống Batch Runner 1.801 runs, nghiên cứu áp lực chọn lọc, Bảng 1).
- **Vinh:** Gradient & Evaluation (Giải tích Soft Macro-F1, Adam, L-BFGS-B, 5-Fold CV chọn λ*, cổng test_gate, Separable Oracle, Bảng 2, Bảng 3, kiểm định MWU-Holm).

---

## 1. Ba câu chữ bắt buộc (Locked Statements)
1. **Quy tắc Neutral độc quyền:** Khi `neutral ∈ S_i` và `|S_i| > 1`: nếu `P[i, neutral] > max P[i, c]` với mọi `c ∈ S_i \ {neutral}` thì giữ Neutral và loại mọi cảm xúc khác; ngược lại loại Neutral và giữ các cảm xúc còn lại (bằng nhau thì loại Neutral).
2. **Phạt L2:** Số hạng phạt L2 đóng vai trò chống quá khớp (regularization), kéo nghiệm về `t_g`. Sự ghép cặp (coupling) giữa các nhãn đến hoàn toàn từ cơ chế Neutral độc quyền và Fallback.
3. **Mô hình duy nhất:** Dự án chỉ sử dụng một mô hình nền duy nhất là ViSoBERT.

---

## 2. Cấu trúc thư mục làm việc chuẩn

```text
ds106-threshold-opt/
├── 00_START_HERE/                     # Hợp đồng, quy tắc nộp, PREREG, môi trường
├── 01_TASKS/                          # Phân rã nhiệm vụ chi tiết 4 tuần của 4 thành viên
│   ├── Week1/  Week2/  Week3/  Week4/
├── 02_SAMPLE_DATA/                    # Dữ liệu mock phục vụ unit test và smoke test
├── 03_SUBMISSIONS/                    # Không gian nộp bằng chứng và báo cáo của từng thành viên
│   ├── Ngoc/   Thy/   Nga/   Vinh/
├── src/                               # Kho mã nguồn dùng chung (merge qua PR)
│   ├── contract/                      # Interface hợp đồng, NFETracker, BaseOptimizer, constants
│   ├── core/                          # Nạp dữ liệu, hàm quyết định, hàm mục tiêu, địa hình
│   ├── optimizers/                    # 10 thuật toán tối ưu (local, global_, gradient)
│   ├── runner/                        # Batch runner đa tiến trình, registry, validate_results
│   ├── eval/                          # Cổng Test, separable oracle, đánh giá tổng quát hóa, ablation, stats
│   ├── configs/                       # Cấu hình siêu tham số và các file mapping
│   └── data/                          # Ma trận xác suất thật (nằm ngoài git)
├── outputs/                           # Bảng kết quả benchmark, biểu đồ, quỹ đạo nghiệm
├── artifacts/                         # Nghiệm tối ưu mốc, lambda_star.json, LOCK_FREEZE.json
├── tests/                             # Toàn bộ test case tự động (pytest)
├── scripts/                           # Script tiện ích kiểm toán NFE, kiểm tra cách ly Test
└── reports/                           # Các báo cáo nghiệm thu kỹ thuật và phân tích chuyên sâu