# QUY TẮC NỘP BÀI VÀ QUY TRÌNH DUYỆT PULL REQUEST (PR)

## 1. Nguyên tắc cách ly tập Test
- Tuyệt đối cấm import, đọc hoặc mở file chứa chuỗi `P_test`, `Y_test`, `_test.npy` trong bất kỳ module nào ngoại trừ `src/eval/test_gate.py`.
- Khi xuất hoặc nạp ma trận động trong script tiền xử lý, phải dùng tên sinh từ biến (ví dụ `f"P_{split}.npy"`).

## 2. Checklist bắt buộc khi tạo PR (Phải tick đủ trước khi review)
- [ ] Khóa Interface: Chữ ký hàm và kiểu dữ liệu trả về khớp 100% với hợp đồng kỹ thuật.
- [ ] Kế thừa chuẩn `BaseOptimizer`: Chỉ cài đặt `_search`, tuyệt đối không override phương thức `optimize()`.
- [ ] Không lọt NFE: Mọi lượt đánh giá hàm mục tiêu đều thông qua đối tượng `tracker(theta)`. Không module nào tự ý gọi hàm mục tiêu gốc `make_objective`.
- [ ] Cách ly Test: Lệnh CI `grep -RIn "P_test\|Y_test\|_test\.npy" src --include=*.py` không báo lỗi vi phạm.
- [ ] Tái lập từng bit: Khởi tạo RNG độc lập qua `np.random.default_rng([seed, self.D, zlib.crc32(self.name.encode()), 1])`. Cấm dùng `np.random.seed` toàn cục và hàm `hash()` của Python.
- [ ] Kiểm thử đơn vị: Chạy `pytest` đạt 100% PASS trên môi trường cục bộ.