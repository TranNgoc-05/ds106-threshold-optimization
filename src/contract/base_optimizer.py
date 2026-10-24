from __future__ import annotations
import time, zlib
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from .nfe_tracker import NFETracker, BudgetExhausted, LO, HI

def init_theta(seed: int, D: int, bounds=(LO, HI)) -> np.ndarray:
    """Điểm khởi tạo dùng chung (LD-5)."""
    return np.random.default_rng([seed, D, 0]).uniform(bounds[0], bounds[1], size=D)

class BaseOptimizer(ABC):
    name: str = "base"                       # snake_case, phải khớp registry (mục 2.8)

    def __init__(self, D: int, bounds=(LO, HI), **hparams):
        self.D, self.bounds, self.hparams = int(D), bounds, hparams

    @abstractmethod
    def _search(self, tracker: NFETracker, x0: np.ndarray, rng: np.random.Generator) -> bool:
        """Chạy tìm kiếm, mọi đánh giá đi qua `tracker`. Trả True nếu DỪNG vì tiêu chí hội tụ
        nội bộ trước khi hết budget, False nếu chạy đến khi hết budget."""

    def optimize(self, tracker: NFETracker, seed: int) -> Tuple[np.ndarray, float, dict]:
        rng = np.random.default_rng([seed, self.D, zlib.crc32(self.name.encode()), 1])
        x0 = init_theta(seed, self.D, self.bounds)
        t0 = time.perf_counter()
        try:
            conv = bool(self._search(tracker, x0, rng))
        except BudgetExhausted:
            conv = False
        elapsed = time.perf_counter() - t0
        if tracker.best_theta is None:
            raise RuntimeError("optimizer made no discrete evaluation")
        traj = tracker.trajectory()
        bsf = traj["best_so_far"]
        tol = 0.01 * abs(tracker.best_score)
        hit = np.where(~np.isnan(bsf) & (bsf >= tracker.best_score - tol))[0]
        traj.update(
            actual_nfe=tracker.nfe,
            nfe_budget=tracker.max_nfe,
            is_converged=bool(conv and tracker.remaining > 0),
            time_seconds=elapsed,
            nfe_to_99pct=int(hit[0]) + 1 if hit.size else None,   # NFE đầu tiên đạt 99% giá trị cuối
            extra=getattr(self, "extra", {}),                      # ví dụ diversity theo thế hệ
        )
        return tracker.best_theta.copy(), float(tracker.best_score), traj