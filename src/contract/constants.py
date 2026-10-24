from .nfe_tracker import LO, HI, OFFICIAL_BUDGETS

N_LABELS = 28
NEUTRAL_IDX = 27
MAPPING_MODES = ("global", "freq3", "semantic4", "per_label")
D_OF_MODE = {"global": 1, "freq3": 3, "semantic4": 4, "per_label": 28}
FALLBACK_STRATEGIES = ("argmax", "neutral", "head_argmax", "none")
BENCHMARK_DIMS = (1, 3, 28)
BENCHMARK_SEEDS = tuple(range(20))
TUNING_SEEDS = tuple(range(100, 105))
LANDSCAPE_SEEDS = tuple(range(200, 400))
ALGO_REGISTRY_NAMES = (
    "constant_0.5", "random_search", "coordinate_ascent", "nelder_mead", "powell",
    "simulated_annealing", "differential_evolution", "pso", "optuna_tpe",
    "adam_surrogate", "lbfgsb_surrogate",
)
CSV_COLUMNS = (
    "algorithm_name", "dimension_D", "seed", "nfe_budget",
    "actual_nfe", "best_val_score", "time_seconds", "is_converged"
)
EXPECTED_ROWS_FULL = 10 * len(BENCHMARK_DIMS) * len(BENCHMARK_SEEDS) * len(OFFICIAL_BUDGETS) + 1  # = 1801