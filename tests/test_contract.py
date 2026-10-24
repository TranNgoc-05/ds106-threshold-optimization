import numpy as np
import pytest

from src.contract.base_optimizer import BaseOptimizer, init_theta
from src.contract.constants import (
    ALGO_REGISTRY_NAMES, D_OF_MODE, EXPECTED_ROWS_FULL, HI, LO,
    MAPPING_MODES, OFFICIAL_BUDGETS
)
from src.contract.nfe_tracker import BudgetExhausted, NFETracker


class CountingObjective:
    """Bộ đếm độc lập, nằm ngoài NFETracker."""
    def __init__(self):
        self.calls = 0

    def __call__(self, theta):
        self.calls += 1
        return float(-np.sum((theta - 0.6) ** 2))


class RandomSearchStub(BaseOptimizer):
    name = "random_search"

    def _search(self, tracker, x0, rng):
        while True:
            tracker(rng.uniform(LO, HI, size=self.D))


class EarlyStopStub(BaseOptimizer):
    name = "early_stub"

    def _search(self, tracker, x0, rng):
        for _ in range(7):
            tracker(x0)
        return True


@pytest.mark.parametrize("budget", OFFICIAL_BUDGETS)
def test_tracker_stops_exactly_at_budget(budget):
    obj = CountingObjective()
    tracker = NFETracker(obj, budget)
    for _ in range(budget):
        tracker(np.full(3, 0.5))
    with pytest.raises(BudgetExhausted):
        tracker(np.full(3, 0.5))
    assert obj.calls == budget
    assert tracker.nfe == budget and tracker.remaining == 0


def test_tracker_rejects_unofficial_budget():
    with pytest.raises(ValueError):
        NFETracker(CountingObjective(), 300)
    NFETracker(CountingObjective(), 300, enforce_official=False)


def test_tracker_clips_into_bounds_and_strict_best():
    seen = []
    tracker = NFETracker(lambda th: (seen.append(th.copy()) or 1.0), 200)
    tracker(np.array([-5.0, 9.0]))
    assert np.all(seen[0] >= LO) and np.all(seen[0] <= HI)
    first_best = tracker.best_theta.copy()
    tracker(np.array([0.5, 0.5]))  # cùng điểm số 1.0, không được thay best (strict >)
    assert np.array_equal(tracker.best_theta, first_best)


def test_surrogate_call_costs_one_nfe_and_does_not_set_best():
    tracker = NFETracker(lambda th: 0.5, 200, surrogate=lambda th, tau: (0.9, np.zeros_like(th)))
    tracker(np.full(3, 0.5))
    tracker.evaluate_surrogate(np.full(3, 0.5), 0.1)
    traj = tracker.trajectory()
    assert tracker.nfe == 2 and tracker.best_score == 0.5
    assert list(traj["call_kind"]) == [0, 1]


def test_early_stop_reports_converged():
    tracker = NFETracker(CountingObjective(), 200)
    _, _, traj = EarlyStopStub(D=3).optimize(tracker, seed=0)
    assert traj["is_converged"] is True
    assert traj["actual_nfe"] == 7 < traj["nfe_budget"] == 200


def test_budget_exhausted_is_not_converged():
    tracker = NFETracker(CountingObjective(), 200)
    _, _, traj = RandomSearchStub(D=3).optimize(tracker, seed=0)
    assert traj["is_converged"] is False and traj["actual_nfe"] == 200


def test_bit_exact_same_seed():
    runs = []
    for _ in range(2):
        tracker = NFETracker(CountingObjective(), 1000)
        theta, score, _ = RandomSearchStub(D=28).optimize(tracker, seed=7)
        runs.append((theta.tobytes(), score))
    assert runs[0] == runs[1]


def test_trajectory_length_matches_actual_nfe():
    obj = CountingObjective()
    tracker = NFETracker(obj, 200)
    _, _, traj = RandomSearchStub(D=3).optimize(tracker, seed=1)
    assert len(traj["best_so_far"]) == len(traj["raw_scores"]) == traj["actual_nfe"] == obj.calls
    for key in ("raw_scores", "best_so_far", "call_kind", "actual_nfe", "nfe_budget",
                "is_converged", "time_seconds", "nfe_to_99pct", "extra"):
        assert key in traj


def test_init_theta_deterministic_and_in_bounds():
    a, b = init_theta(3, 28), init_theta(3, 28)
    assert np.array_equal(a, b) and a.shape == (28,) and np.all((a >= LO) & (a <= HI))
    assert not np.array_equal(init_theta(3, 28), init_theta(4, 28))


def test_optimize_returns_locked_tuple():
    out = RandomSearchStub(D=3).optimize(NFETracker(CountingObjective(), 200), seed=0)
    assert len(out) == 3 and isinstance(out[0], np.ndarray) and isinstance(out[1], float) and isinstance(out[2], dict)


def test_constants_are_consistent():
    assert (LO, HI) == (0.05, 0.95) and OFFICIAL_BUDGETS == (200, 1000, 5000)
    assert set(D_OF_MODE) == set(MAPPING_MODES)
    assert len(ALGO_REGISTRY_NAMES) == 11 and len(set(ALGO_REGISTRY_NAMES)) == 11
    assert EXPECTED_ROWS_FULL == 1801