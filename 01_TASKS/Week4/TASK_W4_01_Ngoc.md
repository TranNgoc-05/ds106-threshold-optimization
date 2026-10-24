# TASK 01 — Ngọc — Tuần 4: Đánh giá Test, thẩm định thống kê (G5) và hoàn thiện báo cáo, slide bảo vệ

**Vai trò:** Technical Project Manager & System Architect  
**Owner:** Ngọc  
**Reviewers:** Thy (Core / Local), Nga (Global / Runner), Vinh (Surrogate / Eval)  
**Deadline tuần:** D+27 23:30  
**Mốc trong tuần:** M5 (Bảng 2, Bảng 3, MWU-Holm, đủ dữ liệu) D+24 23:30 · Cổng G5 D+26 23:30 · M6 (báo cáo toàn văn và slide bảo vệ) D+27 23:30

> **Quy ước thời gian.** D0 là ngày đầu Tuần 1; Tuần 4 gồm D+21 đến D+27. Hạn nộp chính thức theo yêu cầu của Bộ môn có thể sớm hơn D+27; nếu vậy Ngọc lùi toàn bộ mốc tương ứng và ghi vào `CHANGE_REQUESTS.md`.
>
> **Phân vai.** Vinh chạy `generalization.py`, `ablation.py`, `stats_mwu_holm.py`. Thy cung cấp Bảng 4 (khung, `landscape`). Nga cung cấp Bảng 1 và biểu đồ benchmark.

## 1. Mục tiêu tuần (Weekly Goal)

1. Giám sát đánh giá độc lập trên tập Test qua `test_gate.py`, bảo đảm Test chỉ bị đọc một lần có kiểm soát sau khi `LOCK_FREEZE.json` đã niêm phong.
2. Đối soát độc lập Bảng 2 (tổng quát hóa), Bảng 3 (Ablation) và `stats_mwu_holm.csv` của Vinh.
3. Cổng G5: kết luận có bằng chứng cho H1–H4 và NFLT đúng theo `PREREG.md`, kể cả kết quả trái dự đoán.
4. Hoàn thiện báo cáo toàn văn `DS106_final_report` và bộ slide bảo vệ (M6).

## 2. Input đầu vào

| Nguồn | Sản phẩm | Đường dẫn |
|---|---|---|
| Vinh | Bảng 2 và bảng phân tán theo thuật toán | `outputs/tables/table2_generalization.csv`, `test_generalization_by_algo.csv` |
| Vinh | Bảng 3 Ablation (8 cấu hình × 2 thuật toán × 20 seeds) | `outputs/tables/table3_ablation.csv`, `outputs/ablation_results.csv` |
| Vinh | Kiểm định MWU-Holm và tổng hợp NFLT | `outputs/tables/stats_mwu_holm.csv`, `nflt_summary.csv` |
| Vinh | Phân tích chuyên sâu | `reports/eval_findings.md`, `reports/surrogate_notes.md` |
| Vinh | Các `θ*` đã chọn | `src/artifacts/theta_frozen/*.json` |
| Thy | Khung và dữ liệu nền của Bảng 4 | `outputs/tables/per_label_table4.csv` (Vinh điền `t_c*`, F1 sau tối ưu) |
| Nga | Bảng 1 và biểu đồ benchmark | `table1_val_performance.csv`, `convergence_curves_D{1,3,28}.png`, `nfe_to_99pct_boxplot.png`, `runtime_vs_nfe.png`, `premature_convergence_de_pso.png` |
| Ngọc (Tuần 3) | Niêm phong và khung báo cáo | `LOCK_FREEZE.json`, `chapter6_analysis.md`, `DS106_final_report.md` |

## 3. Nhiệm vụ kỹ thuật chi tiết

### 3.1 Kiểm tra cách ly Test trước và trong khi đánh giá (D+21 → D+22)

```bash
bash scripts/check_test_isolation.sh        # chỉ test_gate.py và test_test_gate.py được chứa chuỗi Test
python scripts/verify_lock_freeze.py        # hash còn khớp LOCK_FREEZE.json
git diff --stat g4-freeze -- outputs/benchmark_results.csv outputs/benchmark_theta_log.jsonl \
  src/configs/hparams_frozen.yaml src/artifacts/lambda_star.json    # phải rỗng
git log --oneline -- src/eval/test_gate.py   # mọi thay đổi cổng đều có PR và review
```

Yêu cầu bổ sung (ghi vào `SUBMISSION_RULES.md`, không đổi interface): mỗi lần `load_test()` được gọi, Vinh ghi một dòng JSON vào `outputs/test_access_log.jsonl` gồm thời điểm, tên module gọi, sha256 của `LOCK_FREEZE.json`. Ngọc đối chiếu: các lần truy cập chỉ đến từ `generalization.py` và `ablation.py`, không có truy cập từ `cv_lambda.py`, optimizer, hay script tuning. Mọi tuning và chọn `θ*` chỉ dùng Validation; Test chỉ dùng để báo cáo.

### 3.2 Thẩm định Bảng 2 (tổng quát hóa) bằng đường tính độc lập

**Bước 1. Tính lại `θ*` chỉ từ Validation** (không dùng mã của Vinh). Quy tắc: với mỗi cấu hình, `θ*` là nghiệm có `best_val_score` cao nhất trong các lần chạy ở NFE=5000; hòa điểm thì chọn nghiệm gần `t_g` nhất:

```python
import json
import numpy as np
import pandas as pd

res = pd.read_csv("outputs/benchmark_results.csv")
t_g = json.load(open("src/artifacts/t_g.json"))["t_g"]
rows = [json.loads(l) for l in open("outputs/benchmark_theta_log.jsonl", encoding="utf-8")]
theta = pd.DataFrame(rows)[["algorithm_name", "dimension_D", "seed", "nfe_budget", "best_theta"]]
m = res.merge(theta, on=["algorithm_name", "dimension_D", "seed", "nfe_budget"])
m = m[(m["nfe_budget"] == 5000) & (m["algorithm_name"] != "constant_0.5")]

def pick(g):
    top = g[np.isclose(g["best_val_score"], g["best_val_score"].max(), rtol=0, atol=1e-12)]
    dist = top["best_theta"].apply(lambda t: float(np.sum((np.asarray(t) - t_g) ** 2)))
    return top.loc[dist.idxmin()]

best = {int(D): pick(g) for D, g in m.groupby("dimension_D")}
for D, r in best.items():
    print(D, r["algorithm_name"], int(r["seed"]), round(float(r["best_val_score"]), 6))
    frozen = json.load(open(f"src/artifacts/theta_frozen/D{D}.json"))   # điều chỉnh tên file theo Vinh
    assert np.allclose(frozen["theta"], r["best_theta"], atol=1e-12), f"θ* của D={D} lệch bản đóng băng"
```

**Bước 2. Tính Macro-F1 Val/Test, Gap và Weighted-F1 bằng code độc lập, dùng đúng cổng.** Biến dữ liệu Test được đặt tên `P_t`, `Y_t` để không vi phạm quy tắc grep; script nằm trong `scripts/` và chỉ nạp Test qua `test_gate`. Nếu chữ ký `load_test()` của Vinh khác dòng đầu, chỉnh đúng một dòng đó:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.core.decision import predict_labels
from src.core.objective import macro_f1
from src.eval.test_gate import load_test

P_v, Y_v = np.load("src/data/P_val.npy"), np.load("src/data/Y_val.npy")
P_t, Y_t = load_test()                                    # qua cổng; yêu cầu LOCK_FREEZE.json hợp lệ
mapping = {1: "global", 3: "freq3", 28: "per_label"}

def evaluate(theta, D):
    out = {}
    for tag, (P, Y) in {"val": (P_v, Y_v), "test": (P_t, Y_t)}.items():
        pred = predict_labels(P, np.asarray(theta), mapping[D], "argmax", True)
        out[tag] = macro_f1(Y, pred)
        assert abs(out[tag] - f1_score(Y, pred, average="macro", zero_division=0)) < 1e-9
        if tag == "test":
            out["weighted_test"] = f1_score(Y, pred, average="weighted", zero_division=0)
    out["gap"] = out["val"] - out["test"]                 # Val − Test, KHÔNG có số hạng phạt
    return out

t2 = pd.read_csv("outputs/tables/table2_generalization.csv")
# Với từng dòng của t2 (cấu hình, D, θ*): gọi evaluate(theta, D) và so các cột
# Macro-F1 (Val), Macro-F1 (Test), Generalization Gap, Weighted-F1 (Test) với sai số 1e-9.
# Cột "F1 nhóm 5 nhãn yếu nhất": nhóm 5 nhãn được xác định trên Validation @0.5, cố định cho mọi cấu hình.
```

**Điểm Ngọc phải kiểm trong Bảng 2:**

1. Gap là `Val − Test` Macro-F1 **không phạt** (khác với `J` có phạt); cột nào dùng `J` có phạt thì trả lại.
2. Hàng cấu hình gồm: ngưỡng 0.5, D=1, D=3 (`freq3`), D=3÷4 (nhóm ngữ nghĩa `semantic4`), D=28 với `λ=0`, D=28 với `λ=λ*`. **Không có cột CafeBERT** (CR-002).
3. Bảng phân tán `test_generalization_by_algo.csv` có đủ 20 nghiệm mỗi thuật toán × D ở NFE=5000 (mean ± std và gap), để thấy độ ổn định chứ không chỉ nghiệm tốt nhất.
4. Bảng 4 hoàn thiện (28 nhãn): nhóm Head/Mid/Tail đúng `mapping_freq3.json`; `t_c*` lấy từ `θ*` của D=28; F1 trước và sau tối ưu tính lại được từ `θ*`.

### 3.3 Đối soát Ablation Bảng 3

Bảng 3 gồm các `config_id`: `full`, `no_fallback`, `fb_neutral`, `fb_head`, `no_excl_neutral`, `lam0`, so sánh `freq3` với `semantic4`, và `sep_check`; chạy bằng hai thuật toán `coordinate_ascent` và `differential_evolution` ở NFE=5000, seeds 0–19.

| Kiểm tra | Cách làm | Điều kiện đạt |
|---|---|---|
| Đủ số dòng | Đếm theo `config_id` | Mỗi `config_id` có 2 thuật toán × 20 seeds = 40 dòng |
| Tham số đúng cấu hình | Đối chiếu tham số lưu kèm với bảng ở EVAL-01 mục E | `fallback`, `excl`, `λ` khớp; `t_g` giữ cố định cho mọi cấu hình |
| Nhất quán với benchmark chính | Với `config_id = full`: so `best_val_score` với dòng cùng khóa `(algorithm_name, D=28, seed, 5000)` trong `benchmark_results.csv` | Bằng nhau từng bit (cùng hàm mục tiêu và cùng siêu tham số) |
| Quy tắc Neutral mới | `no_excl_neutral` thực sự bỏ bước so sánh xác suất; `full` dùng quy tắc theo CR-003 | Có test và log chứng minh; không còn mã "luôn loại Neutral" |
| Ground truth tách biến | `sep_check`: `optimality_gap = J* − best_val_score` | Không có giá trị `< −1e-9`; có `separable_oracle` khớp brute-force |
| Diễn giải coupling | Đọc phần nhận xét của Bảng 3 | Coupling quy về Neutral độc quyền và Fallback; phạt L2 mô tả là regularization (CR-001); `lam0` so với `full` chỉ dùng để nói về quá khớp |

### 3.4 Đối soát kiểm định MWU-Holm

Quy mô kỳ vọng của `stats_mwu_holm.csv`: 9 ô (D, budget), mỗi ô có 9 so sánh `primary` (mỗi thuật toán với `random_search`) và 45 so sánh `exploratory`, tức 81 + 405 = 486 dòng. Script độc lập `scripts/check_stats.py` tính lại ba ô ngẫu nhiên:

```python
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

bench = pd.read_csv("outputs/benchmark_results.csv")
bench = bench[bench["algorithm_name"] != "constant_0.5"]
stats = pd.read_csv("outputs/tables/stats_mwu_holm.csv")
assert (stats["family"] == "primary").sum() == 81 and (stats["family"] == "exploratory").sum() == 405

rng = np.random.default_rng(106)
cells = [(D, B) for D in (1, 3, 28) for B in (200, 1000, 5000)]
for D, B in [cells[i] for i in rng.choice(len(cells), size=3, replace=False)]:
    cell = bench[(bench["dimension_D"] == D) & (bench["nfe_budget"] == B)]
    rs = cell[cell["algorithm_name"] == "random_search"].sort_values("seed")["best_val_score"].to_numpy()
    algos = sorted(a for a in cell["algorithm_name"].unique() if a != "random_search")
    p_raw = []
    for a in algos:
        x = cell[cell["algorithm_name"] == a].sort_values("seed")["best_val_score"].to_numpy()
        p_raw.append(1.0 if np.array_equal(np.sort(x), np.sort(rs)) else
                     mannwhitneyu(x, rs, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue)
    p_holm = multipletests(p_raw, method="holm")[1]
    sub = stats[(stats["family"] == "primary") & (stats["dimension_D"] == D) & (stats["nfe_budget"] == B)
                & (stats["algo_b"] == "random_search")].set_index("algo_a")
    for a, ph in zip(algos, p_holm):
        assert abs(sub.loc[a, "p_holm"] - ph) < 1e-9, (D, B, a, sub.loc[a, "p_holm"], ph)
print("MWU-Holm độc lập: KHỚP trên 3 ô ngẫu nhiên")
```

Ngọc kiểm thêm: cỡ hiệu ứng A12 (Vargha–Delaney) và rank-biserial có mặt ở mọi dòng; ô suy biến (`degenerate = True`) được xử lý `p = 1.0`; `constant_0.5` bị loại khỏi kiểm định; `nflt_summary.csv` khớp với việc tự đếm số ô dẫn đầu của từng thuật toán theo trung vị `best_val_score`.

### 3.5 Cổng G5: Trung thực khoa học (D+25 → D+26)

**Bảng quyết định H1–H4 và NFLT.** Tiêu chí lấy nguyên văn từ `PREREG.md`; cột nguồn trỏ vào dòng bảng cụ thể; không đổi tiêu chí sau khi đã thấy số.

| Giả thuyết | Nguồn số liệu | Quy tắc (theo `PREREG.md`) | Kết luận |
|---|---|---|---|
| H1 | `t_g.json`, `meta.json`, `table2_generalization.csv` | `t_g > 0.5`; Macro-F1 tại `t_g` cao hơn tại 0.5; Macro-Precision < Macro-Recall tại 0.5 | Ủng hộ / Không ủng hộ / Không kết luận được |
| H2 | `table2_generalization.csv`, `test_generalization_by_algo.csv` | Trung vị Gap của (D=28, λ=0) lớn hơn Gap của D=3 và của (D=28, λ=λ*), NFE=5000 | Ủng hộ / Không ủng hộ / Không kết luận được |
| H3 | `stats_mwu_holm.csv`, `nflt_summary.csv`, `optimality_gap_D28.csv` | Ở D=28, NFE=5000, `coordinate_ascent` không kém có ý nghĩa so với DE và PSO; và DE, PSO hoặc TPE dẫn đầu ít nhất một ô thuộc tập đã đăng ký | Ủng hộ / Không ủng hộ / Không kết luận được |
| H4 | `benchmark_results.csv` (`nfe_to_99pct`), `stats_mwu_holm.csv`, `surrogate_notes.md` | NFE=5000, D ∈ {3, 28}: trung vị `nfe_to_99pct` của gradient nhỏ hơn mọi hộp đen **và** trung vị `best_val_score` thấp hơn hộp đen tốt nhất với p < 0.05 | Ủng hộ / Không ủng hộ / Không kết luận được |
| NFLT | `nflt_summary.csv` | ≥ 2 thuật toán khác nhau giữ hạng nhất trong 9 ô **và** `random_search` không đứng cuối ở ít nhất một ô | Ủng hộ / Không ủng hộ / Không kết luận được |

**Sổ cái bằng chứng (`reports/G5_claims_ledger.csv`).** Mọi con số và nhận định trong báo cáo, slide và `eval_findings.md` có một dòng với các cột `claim_id, claim_text, source_file, filter, column, value, tolerance`. Script kiểm tra:

```python
import sys
import pandas as pd

ledger = pd.read_csv("reports/G5_claims_ledger.csv")
bad = 0
for _, c in ledger.iterrows():
    df = pd.read_csv(c["source_file"])
    row = df.query(c["filter"])
    ok = len(row) == 1 and abs(float(row.iloc[0][c["column"]]) - float(c["value"])) <= float(c["tolerance"])
    bad += not ok
    print(c["claim_id"], "ĐẠT" if ok else "LỆCH / KHÔNG DUY NHẤT")
sys.exit(1 if bad else 0)
```

**Điều kiện đạt G5.** (1) `check_claims` ĐẠT toàn bộ; (2) mỗi giả thuyết có một dòng kết luận đúng một trong ba nhãn ở bảng trên; (3) giả thuyết không được ủng hộ vẫn giữ nguyên số liệu và có đoạn biện luận dựa trên lý thuyết môn học (Chapter 4, 5, 6); (4) không có đoạn nào trong báo cáo gán coupling cho phạt L2; (5) nhận xét về surrogate nêu rõ nó không mô hình hóa Neutral độc quyền và Fallback (hai cơ chế không khả vi) bên cạnh việc `τ` làm phẳng bậc thang; kết luận Bảng 2 có đủ hai chiều: nghiệm tốt nhất và độ phân tán theo seed.

### 3.6 Hoàn thiện báo cáo toàn văn `DS106_final_report` (D+22 → D+27)

Cấu trúc cuối cùng (Ngọc viết các phần in đậm; các phần còn lại do thành viên cung cấp dữ liệu, Ngọc biên tập và dẫn nguồn):

| Phần | Nội dung | Nguồn bảng/biểu đồ |
|---|---|---|
| **1. Đặt vấn đề, 2. Bối cảnh, 3. Cơ sở lý thuyết** | Đã có từ Tuần 3, rà soát lại sau khi có số liệu | Slide DS106 Chapter 1–6 |
| **4. Mô hình hóa và giao thức** | `J_λ`, quy tắc Neutral, Fallback, surrogate, quy tắc đếm NFE, đăng ký trước, cô lập Test | `PREREG.md`, `CONTRACT_v1.0.md` |
| 5. Khảo sát địa hình | 1D, 2D, plateau ratio, số cực trị địa phương | `landscape_summary.csv` |
| 6. Kết quả benchmark | Bảng 1; đường cong hội tụ; chi phí thời gian | `table1_val_performance.csv`, biểu đồ |
| **7. Phân tích hội tụ sớm** | DE/PSO, áp lực chọn lọc; hội tụ sớm của nhóm local và gradient | `selective_pressure_study.csv`, `premature_convergence_local.csv` |
| 8. Tổng quát hóa và Ablation | Bảng 2, Bảng 3, Bảng 4; optimality gap | `table2_generalization.csv`, `table3_ablation.csv`, `per_label_table4.csv`, `optimality_gap_D28.csv` |
| 9. Kiểm định thống kê | MWU-Holm, A12; tổng hợp NFLT | `stats_mwu_holm.csv`, `nflt_summary.csv` |
| **10. Thảo luận** | Trả lời H1–H4 và NFLT; biện luận nguyên nhân toán học (hàm bậc thang, plateau, tách biến gần đúng, áp lực chọn lọc) | Bảng quyết định ở 3.5 |
| **11. Hạn chế** | Chỉ một mô hình (ViSoBERT) và một tập dữ liệu; nghiệm chọn theo Validation; hàm bậc thang làm hòa điểm nhiều nên đã khóa luật tie-break; chi phí gradient (value + gradient) được tính 1 NFE; chọn 20 seeds; phạm vi tuning ≤ 5 cấu hình; surrogate không mô hình hóa Neutral và Fallback | Toàn bộ bảng |
| **12. Kết luận** | Đóng góp theo góc nhìn tối ưu hóa ứng dụng; hướng phát triển | |
| 13. Tài liệu tham khảo | Chuẩn hóa một kiểu trích dẫn duy nhất | Danh mục dưới đây |

**Tài liệu tham khảo bắt buộc:** Nguyễn-Tất, B. T. (2026). *Slide bài giảng môn Tối ưu hóa ứng dụng (DS106)*, UIT, ĐHQG-HCM: Chapter 1 *Function Optimization* (pp. 1–10); Chapter 2 *Optimization and Machine Learning* (pp. 1–18); Chapter 3 *Optimization Algorithm Selection* (pp. 1–15); Chapter 4 *No Free Lunch Theorem* (pp. 1–8); Chapter 5 *Local and Global Optimization* (pp. 1–12); Chapter 6 *Premature Convergence* (pp. 1–9). Bài báo ViGoEmotions (2025); Demszky et al. (2020), GoEmotions; Nguyen et al. (2023), ViSoBERT; Wolpert & Macready (1997), NFL; Akiba et al. (2019), Optuna; Bergstra et al. (2011), TPE; Storn & Price (1997), DE; Kennedy & Eberhart (1995), PSO; Kirkpatrick et al. (1983), SA; Nelder & Mead (1965); Powell (1964); Kingma & Ba (2015), Adam; Byrd et al. (1995), L-BFGS-B; Mann & Whitney (1947); Holm (1979); Vargha & Delaney (2000); Bartz-Beielstein et al. (2020); Hansen et al. (2021); Sechidis et al. (2011). Ngọc kiểm tra từng nguồn có thực sự được trích trong nội dung; nguồn nào chưa tra được bản chính thức thì ghi rõ để đối chiếu với bản gốc trước khi nộp.

### 3.7 Slide bảo vệ (16:9) và tổng duyệt (D+25 → D+27)

**Cấu trúc đề xuất (khoảng 18 slide, 20 phút trình bày và 10 phút Q&A; điều chỉnh theo quy định của Bộ môn):**

| Slide | Nội dung | Người trình bày |
|---|---|---|
| 1–2 | Tên đề tài; bài toán ViGoEmotions, ngưỡng 0.5 và đuôi dài | Ngọc |
| 3–4 | Mô hình hóa tối ưu: `Θ`, `J_λ`, Neutral độc quyền, Fallback; giao thức công bằng (NFE, 20 seeds, cô lập Test) | Ngọc |
| 5–6 | Địa hình hàm số 1D/2D, plateau ratio, số cực trị; nhóm local (CA, NM, Powell) | Thy |
| 7–9 | Nhóm toàn cục (RS, SA, DE, PSO, TPE); Bảng 1; đường cong hội tụ | Nga |
| 10 | Hội tụ sớm và áp lực chọn lọc (Chapter 6) | Nga |
| 11–12 | Surrogate gradient, quy trình 5-fold CV chọn `λ*` | Vinh |
| 13–14 | Bảng 2 (tổng quát hóa) và Bảng 3 (Ablation) | Vinh |
| 15 | MWU-Holm và kiểm chứng NFLT | Vinh |
| 16 | Trả lời H1–H4: kết quả ủng hộ hoặc không ủng hộ | Ngọc |
| 17 | Hạn chế và hướng phát triển | Ngọc |
| 18 | Kết luận và đóng góp theo góc nhìn tối ưu hóa ứng dụng | Ngọc |

**Quy tắc slide:** mỗi con số trích từ một dòng trong sổ cái bằng chứng; biểu đồ lấy trực tiếp từ `outputs/figures/`; không đưa kết quả chưa qua G5; không nhắc CafeBERT như một thực nghiệm đã làm.

**Tổng duyệt và Q&A:** hai buổi tập luyện (D+25, D+26), bấm giờ từng phần, hoán đổi người trình bày một phần để mọi người hiểu toàn bộ. Chuẩn bị tối thiểu 12 câu hỏi phản biện kèm dòng bằng chứng trả lời, gồm: vì sao chọn `freq3` 9/10/9; vì sao tính 1 NFE cho lời gọi surrogate cả gradient; tie-break trên plateau ảnh hưởng thế nào; vì sao Random Search là baseline và NFLT được kiểm chứng ra sao; vì sao `lbfgsb_surrogate` dừng sớm; vì sao phạt L2 không tạo coupling; vì sao chỉ dùng ViSoBERT; nghiệm có tổng quát sang tập Test không; ảnh hưởng của hàm bậc thang tới Nelder-Mead; vì sao dùng MWU-Holm thay vì ANOVA; kết luận nào trái dự đoán và biện luận ra sao.

### 3.8 Chốt kho mã nguồn và nộp bài (D+27)

```bash
bash scripts/check_test_isolation.sh && pytest -q
python scripts/verify_lock_freeze.py && python scripts/check_stats.py && python scripts/check_claims.py
git status --porcelain      # phải rỗng
git tag v1.0-final
```

`README.md` ở gốc repo phải có: mục tiêu, cấu trúc, môi trường (Python 3.11, `requirements.txt`, biến `OMP_NUM_THREADS=1`), thứ tự lệnh tái lập (`make_mock_data.py` → `train_visobert.py` → `landscape` → `run_benchmark` → `validate_results` → `make_lock_freeze` → `generalization` → `ablation` → `stats_mwu_holm`), hash dữ liệu, và ba câu chữ bắt buộc ở Tuần 1.

## 4. Output tuần

| Nhóm | File |
|---|---|
| Kiểm toán cuối | `scripts/check_stats.py`, `scripts/check_claims.py`, `reports/G5_claims_ledger.csv`, `reports/audit_G5_scientific_integrity.md` |
| Báo cáo | `reports/DS106_final_report.docx` (hoặc PDF theo yêu cầu của Bộ môn) |
| Slide | Bộ slide bảo vệ 16:9, kèm file kịch bản Q&A |
| Kho mã | `README.md` hoàn chỉnh, tag `v1.0-final` |

## 5. Definition of Done (DoD) Tuần 4

- [ ] `check_test_isolation.sh` sạch; `verify_lock_freeze.py` KHỚP; `test_access_log.jsonl` chỉ có truy cập từ `generalization.py` và `ablation.py`.
- [ ] Bảng 2 tính lại độc lập khớp bảng của Vinh ở sai số `1e-9`; Gap là `Val − Test` Macro-F1 không phạt; không còn cột hay thành phần CafeBERT.
- [ ] Bảng 3: `full` khớp từng bit với benchmark chính; `sep_check` không có gap `< −1e-9`; nhận xét coupling quy về Neutral độc quyền và Fallback.
- [ ] `stats_mwu_holm.csv` có đủ 486 dòng; ba ô tính lại độc lập khớp `p_holm` ở `1e-9`.
- [ ] **Cổng G5:** sổ cái bằng chứng ĐẠT toàn bộ; mỗi giả thuyết H1–H4 và NFLT có kết luận theo đúng tiêu chí đã đăng ký; giả thuyết trái dự đoán vẫn giữ nguyên số liệu.
- [ ] Bảng 1, 2, 3, 4 điền đầy đủ 100% số liệu thật, không còn ô trống.
- [ ] Báo cáo toàn văn và slide được nộp đúng hạn quy định của môn học; nhóm đã tổng duyệt ít nhất hai lần.