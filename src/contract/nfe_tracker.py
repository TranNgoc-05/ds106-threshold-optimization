from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple

LO, HI = 0.05, 0.95
OFFICIAL_BUDGETS = (200, 1000, 5000)

class BudgetExhausted(Exception):
    """Ném TRƯỚC khi đánh giá lần gọi thứ max_nfe+1."""

class NFETracker:
    def __init__(self, objective: Callable[[np.ndarray], float], max_nfe: int,
                 surrogate: Optional[Callable[[np.ndarray, float], Tuple[float, np.ndarray]]] = None,
                 bounds=(LO, HI), enforce_official: bool = True):
        if enforce_official and max_nfe not in OFFICIAL_BUDGETS:
            raise ValueError(f"max_nfe must be in {OFFICIAL_BUDGETS}")
        self.objective, self.surrogate = objective, surrogate
        self.max_nfe, self.lo, self.hi = int(max_nfe), bounds[0], bounds[1]
        self.nfe = 0
        self.best_score, self.best_theta = -np.inf, None
        self._raw, self._best, self._kind = [], [], []

    @property
    def remaining(self) -> int:
        return self.max_nfe - self.nfe

    def _charge(self):
        if self.nfe >= self.max_nfe:
            raise BudgetExhausted()
        self.nfe += 1

    def _clip(self, theta):
        return np.clip(np.asarray(theta, dtype=np.float64).ravel(), self.lo, self.hi)

    def __call__(self, theta) -> float:               # đánh giá J_λ rời rạc, 1 NFE
        self._charge()
        th = self._clip(theta)                        # tracker luôn clip vào [0.05, 0.95]
        s = float(self.objective(th))
        if s > self.best_score:                       # strict >: giữ nghiệm gặp đầu tiên
            self.best_score, self.best_theta = s, th.copy()
        self._raw.append(s); self._best.append(self.best_score); self._kind.append(0)
        return s

    def evaluate_surrogate(self, theta, tau: float):  # surrogate + gradient, 1 NFE
        if self.surrogate is None:
            raise RuntimeError("tracker was built without surrogate")
        self._charge()
        th = self._clip(theta)
        val, grad = self.surrogate(th, tau)
        self._raw.append(np.nan)
        self._best.append(self.best_score if self.best_theta is not None else np.nan)
        self._kind.append(1)
        return float(val), np.asarray(grad, dtype=np.float64)

    def trajectory(self) -> dict:
        return {
            "raw_scores": np.asarray(self._raw, dtype=np.float64),      # NaN tại lần gọi surrogate
            "best_so_far": np.asarray(self._best, dtype=np.float64),    # độ dài = actual_nfe
            "call_kind": np.asarray(self._kind, dtype=np.uint8)         # 0 = rời rạc, 1 = surrogate
        }