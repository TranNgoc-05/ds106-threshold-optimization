"""Generate deterministic mock artifacts for the ViGoEmotions threshold-optimization project.

Expected project layout:
    project/
      02_DATA/make_mock_data.py
      data_reannotation/train_reannotation.csv
      configs/mappings/mapping_semantic4.json
      src/contract/constants.py

Run from any working directory:
    python 02_DATA/make_mock_data.py
Optional:
    python 02_DATA/make_mock_data.py --train-csv data_reannotation/train_reannotation.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

# Resolve paths from this script, not from the caller's current directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAIN_CSV = SCRIPT_DIR / "data_reannotation" / "train_reannotation.csv"
DEFAULT_SEMANTIC_MAPPING = PROJECT_ROOT / "src" / "configs" / "mappings" / "mapping_semantic4.json"

SEED = 106
N_LABELS = 28
NEUTRAL_IDX = 27
LABEL_NAMES = [
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
    "pride", "admiration", "gratitude", "relief", "approval", "realization",
    "surprise", "curiosity", "confusion", "fear", "nervousness", "remorse",
    "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger",
    "annoyance", "disapproval", "neutral",
]
GROUP_NAMES = ["Head", "Mid", "Tail"]
SEMANTIC_GROUPS = ["pos", "amb", "neg", "neu"]

# Parameters are deliberately label-dependent so Tail is harder than Head.
A_HEAD = 1.8
A_TAIL = 1.15
BIAS_B = 0.8
SIGMA = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_constants_neutral_idx() -> int:
    """Load NEUTRAL_IDX from the project's constants.py when importable."""
    import sys

    src_dir = PROJECT_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from contract.constants import NEUTRAL_IDX as idx  # type: ignore
        return int(idx)
    except Exception:
        # The supplied constants.py uses a relative import from .nfe_tracker;
        # if package imports are unavailable, retain the explicitly supplied contract value.
        return NEUTRAL_IDX


def _parse_label_cell(value: Any) -> list[str]:
    """Parse a CSV cell containing labels as JSON, Python-like list, or delimited text."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            s = s[1:-1]
    for sep in (";", "|", ","):
        if sep in s:
            return [x.strip().strip("'\"") for x in s.split(sep) if x.strip()]
    return [s.strip("'\"")]


def load_train_counts(csv_path: Path) -> tuple[np.ndarray, str]:
    """Read label counts from a train CSV.

    Supported formats:
    1) 28 binary columns named exactly after LABEL_NAMES;
    2) one label-list column named labels/emotions/label/emotion/labels_json;
    3) a text column with comma/semicolon/pipe-separated label names.

    If the actual CSV uses another schema, this function fails explicitly rather
    than silently fabricating a mapping.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy train CSV: {csv_path}\n"
            "Hãy truyền đúng đường dẫn bằng --train-csv."
        )

    df = pd.read_csv(csv_path)
    normalized = {str(c).strip().lower(): c for c in df.columns}

    # Case 1: one binary column per emotion.
    if all(name in normalized for name in LABEL_NAMES):
        cols = [normalized[name] for name in LABEL_NAMES]
        matrix = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        if not np.isin(matrix, [0, 1]).all():
            raise ValueError("Các cột nhãn dạng one-hot phải chỉ chứa 0/1.")
        counts = matrix.astype(np.uint8).sum(axis=0).astype(np.int64)
        return counts, f"{csv_path}; binary label columns"

    # Case 2: a single multi-label column.
    candidate_names = ("labels", "emotions", "label", "emotion", "labels_json", "emotion_labels")
    label_col = next((normalized[name] for name in candidate_names if name in normalized), None)

    # Case 3: common fallback column names that may contain label lists.
    if label_col is None:
        for name in ("target", "category", "annotations"):
            if name in normalized:
                label_col = normalized[name]
                break

    if label_col is None:
        raise ValueError(
            "Không nhận diện được schema nhãn trong train CSV. "
            f"Các cột hiện có: {list(df.columns)}. "
            "Cần 28 cột nhãn 0/1 theo tên emotion, hoặc một cột labels/emotions "
            "chứa danh sách nhãn."
        )

    name_to_id = {name: i for i, name in enumerate(LABEL_NAMES)}
    counts = np.zeros(N_LABELS, dtype=np.int64)
    unknown: set[str] = set()
    for cell in df[label_col].tolist():
        labels = _parse_label_cell(cell)
        for label in set(labels):
            key = label.strip().lower()
            if key in name_to_id:
                counts[name_to_id[key]] += 1
            elif key.isdigit() and 0 <= int(key) < N_LABELS:
                counts[int(key)] += 1
            elif key:
                unknown.add(label)
    if unknown:
        preview = sorted(unknown)[:12]
        raise ValueError(
            f"Cột '{label_col}' có nhãn chưa khớp 28 emotion names: {preview}. "
            "Kiểm tra schema hoặc chuẩn hóa tên nhãn trước khi chạy."
        )
    if counts.sum() == 0:
        raise ValueError(f"Không đọc được nhãn hợp lệ từ cột '{label_col}'.")
    return counts, f"{csv_path}; label column '{label_col}'"


def make_mock_frequencies(group_of_label: np.ndarray) -> np.ndarray:
    """Assign exponential frequencies by frequency rank, with Head above Tail."""
    freqs = np.zeros(N_LABELS, dtype=np.float64)
    emotion_ids = np.arange(NEUTRAL_IDX)
    # Head/Mid/Tail ordering comes from train_counts; smaller id breaks ties.
    ordered = sorted(
        emotion_ids.tolist(),
        key=lambda idx: (int(group_of_label[idx]), idx),
    )
    emotion_freqs = np.geomspace(0.22, 0.045, N_LABELS - 1)
    for rank, label_id in enumerate(ordered):
        freqs[label_id] = emotion_freqs[rank]
    # Neutral is generated exclusively for rows with no emotion labels.
    return freqs


def make_split(n: int, rng: np.random.Generator, freqs: np.ndarray,
               group_of_label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Y = np.zeros((n, N_LABELS), dtype=np.uint8)
    emotion_freqs = freqs[:NEUTRAL_IDX]
    Y[:, :NEUTRAL_IDX] = (
        rng.random((n, NEUTRAL_IDX)) < emotion_freqs
    ).astype(np.uint8)

    no_emotion = Y[:, :NEUTRAL_IDX].sum(axis=1) == 0
    Y[no_emotion, NEUTRAL_IDX] = 1

    # Guarantee at least one positive label; neutral is the exclusive fallback.
    empty = Y.sum(axis=1) == 0
    Y[empty, NEUTRAL_IDX] = 1

    # Label-specific separability: Head larger a, Tail smaller a.
    a_by_label = np.full(N_LABELS, A_TAIL, dtype=np.float64)
    a_by_label[group_of_label == 0] = A_HEAD
    a_by_label[group_of_label == 1] = (A_HEAD + A_TAIL) / 2.0
    a_by_label[NEUTRAL_IDX] = A_HEAD
    logits = a_by_label[None, :] * (2.0 * Y.astype(np.float64) - 1.0)
    logits += BIAS_B + SIGMA * rng.standard_normal((n, N_LABELS))
    P = expit(logits).astype(np.float64)

    if P.shape != (n, N_LABELS) or Y.shape != (n, N_LABELS):
        raise AssertionError("Sai shape dữ liệu mock.")
    if P.dtype != np.float64 or Y.dtype != np.uint8:
        raise AssertionError("Sai dtype dữ liệu mock.")
    if not np.isfinite(P).all() or ((P < 0) | (P > 1)).any():
        raise AssertionError("P chứa giá trị không hợp lệ.")
    if not np.all(Y.sum(axis=1) >= 1):
        raise AssertionError("Có hàng Y không có nhãn dương.")
    if np.any((Y[:, NEUTRAL_IDX] == 1) & (Y[:, :NEUTRAL_IDX].sum(axis=1) > 0)):
        raise AssertionError("Vi phạm exclusive-neutral.")
    return P, Y


def make_freq3_mapping(train_counts: np.ndarray | None,
                       fallback_freqs: np.ndarray,
                       source: str) -> dict[str, Any]:
    counts = train_counts if train_counts is not None else np.rint(fallback_freqs * 1_000_000).astype(np.int64)
    # Stable sort ensures ties are resolved by smaller column index.
    order = np.lexsort((np.arange(N_LABELS), -counts))
    head_idx = order[:9].astype(int).tolist()
    mid_idx = order[9:19].astype(int).tolist()
    tail_idx = order[19:].astype(int).tolist()
    group = np.empty(N_LABELS, dtype=np.int8)
    group[head_idx] = 0
    group[mid_idx] = 1
    group[tail_idx] = 2
    return {
        "group_of_label": group.tolist(),
        "head_idx": head_idx,
        "mid_idx": mid_idx,
        "tail_idx": tail_idx,
        "train_counts": counts.astype(int).tolist(),
        "label_names": LABEL_NAMES,
        "group_names": GROUP_NAMES,
        "source": source,
        "mock": train_counts is None,
    }


def make_semantic4_mapping(source_path: Path | None) -> dict[str, Any]:
    if source_path is not None and source_path.exists():
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        id_to_group = raw.get("id_to_group", {})
        group_to_idx = {name: i for i, name in enumerate(SEMANTIC_GROUPS)}
        try:
            group = [group_to_idx[id_to_group[str(i)]] for i in range(N_LABELS)]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{source_path} thiếu id_to_group đầy đủ cho 28 nhãn."
            ) from exc
        source = str(source_path)
    else:
        # This fallback matches the mapping supplied by the user in chat.
        group = (
            [0] * 12 + [1] * 4 + [2] * 11 + [3]
        )
        source = "fallback mapping supplied in project conversation; verify against source JSON"
        warnings.warn(
            "Không tìm thấy mapping_semantic4.json tại đường dẫn mặc định; "
            "dùng mapping dự phòng theo nội dung bạn đã cung cấp.",
            RuntimeWarning,
        )
    return {
        "group_of_label": group,
        "group_names": SEMANTIC_GROUPS,
        "label_names": LABEL_NAMES,
        "source": source,
        "mock": True,
    }


def make_cv_folds(Y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Create deterministic 5-fold assignments; prefer multilabel stratification."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        folds = np.empty(len(Y), dtype=np.int8)
        for fold_id, (_, test_idx) in enumerate(splitter.split(np.zeros(len(Y)), Y)):
            folds[test_idx] = fold_id
        return folds
    except ImportError:
        warnings.warn(
            "iterative-stratification chưa cài; dùng KFold fallback (không đảm bảo "
            "cân bằng đa nhãn). Cài bằng: pip install iterative-stratification",
            RuntimeWarning,
        )
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = np.empty(len(Y), dtype=np.int8)
        for fold_id, (_, test_idx) in enumerate(splitter.split(Y)):
            folds[test_idx] = fold_id
        return folds


def golden_cases() -> dict[str, Any]:
    """Golden inputs and expected outcomes described in TASK_02_Thy_Week1 §3.3."""
    def row(**values: float) -> list[float]:
        """Probability row; unspecified labels stay at 0.01."""
        result = [0.01] * N_LABELS
        for name, value in values.items():
            idx = int(name) if name.isdigit() else LABEL_NAMES.index(name)
            result[idx] = float(value)
        return result

    def binary_row(**values: float) -> list[int]:
        """Expected-label row; unspecified labels are 0."""
        result = [0] * N_LABELS
        for name, value in values.items():
            idx = int(name) if name.isdigit() else LABEL_NAMES.index(name)
            result[idx] = int(value)
        return result

    theta = [0.5]
    base = {
        "mapping_mode": "global",
        "fallback_strategy": "argmax",
        "exclusive_neutral": True,
        "theta": theta,
        "notes": "Các cột không nêu có P=0.01; ngưỡng dùng >=.",
    }
    cases = [
        {"id": "G01", **base, "P": [row(neutral=0.70, amusement=0.60)],
         "expected_Y": [binary_row(neutral=1)]},
        {"id": "G02", **base, "P": [row(neutral=0.55, amusement=0.60)],
         "expected_Y": [binary_row(amusement=1)]},
        {"id": "G03", **base, "P": [row(neutral=0.60, amusement=0.60)],
         "expected_Y": [binary_row(amusement=1)]},
        {"id": "G04", **base, "P": [row(neutral=0.80, amusement=0.60, joy=0.55)],
         "expected_Y": [binary_row(neutral=1)]},
        {"id": "G05", **base, "P": [row(neutral=0.70)],
         "expected_Y": [binary_row(neutral=1)]},
        {"id": "G06", **{**base, "exclusive_neutral": False},
         "P": [row(neutral=0.70, amusement=0.60)],
         "expected_Y": [binary_row(neutral=1, amusement=1)]},
        {"id": "G07", **{**base, "fallback_strategy": "argmax"},
         "theta": [0.9], "P": [row(amusement=0.40, joy=0.80)],
         "expected_Y": [binary_row(joy=1)]},
        {"id": "G08", **{**base, "fallback_strategy": "neutral"},
         "P": [row(amusement=0.40, joy=0.30)], "expected_Y": [binary_row(neutral=1)]},
        {"id": "G09", **{**base, "fallback_strategy": "head_argmax"},
         "theta": [0.9], "P": [row(amusement=0.40, joy=0.80)], "expected_Y": None,
         "note": "Tính expected_Y bằng argmax của P giới hạn trong head_idx của mapping_freq3."},
        {"id": "G10", **{**base, "fallback_strategy": "none"},
         "P": [row(amusement=0.40, joy=0.30)], "expected_Y": [[0] * N_LABELS]},
        {"id": "G11", **base, "P": [row(amusement=0.50)],
         "expected_Y": [binary_row(amusement=1)]},
        {"id": "G12", **{**base, "theta": [0.9]},
         "P": [row(amusement=0.80, excitement=0.80)],
         "expected_Y": [binary_row(amusement=1)],
         "note": "Không có nhãn vượt ngưỡng; fallback argmax phá hòa bằng chỉ số nhỏ nhất."},
        {"id": "G13", **{**base, "theta": [0.5, 0.5]}, "P": [row(amusement=0.70)],
         "expected_error": "ValueError"},
        {"id": "G14", **{**base, "theta": [0.99]}, "P": [row(amusement=0.70)],
         "expected_error": "ValueError",
         "note": "Có thể tách thành hai ca con: theta ngoài miền và P > 1."},
        {"id": "G15", "subcases": [
            {"mapping_mode": "global", "D": 1, "theta": [0.4]},
            {"mapping_mode": "freq3", "D": 3, "theta": [0.3, 0.5, 0.7]},
            {"mapping_mode": "semantic4", "D": 4, "theta": [0.3, 0.4, 0.6, 0.7]},
            {"mapping_mode": "per_label", "D": 28, "theta": [0.5] * N_LABELS},
        ], "expected": "Mỗi mode mở rộng theta thành vector 28 chiều đúng theo group_of_label."},
    ]
    return {
        "schema_version": 1,
        "n_labels": N_LABELS,
        "neutral_idx": NEUTRAL_IDX,
        "threshold_rule": "P >= theta",
        "cases": cases,
        "warning": (
            "Golden cases là dữ liệu đặc tả tham khảo; hãy đối chiếu tên trường với "
            "schema thực tế của tests/test_decision.py trước khi dùng tự động."
        ),
    }


def build(name: str, n: int, train_counts: np.ndarray | None,
          train_source: str, semantic_path: Path | None, neutral_idx: int) -> dict[str, str]:
    out_dir = SCRIPT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = out_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    provisional_freqs = np.zeros(N_LABELS, dtype=np.float64)
    provisional_freqs[:NEUTRAL_IDX] = np.geomspace(
        0.22, 0.045, N_LABELS - 1
    )

    freq3 = make_freq3_mapping(
        train_counts,
        provisional_freqs,
        train_source,
    )

    sem4 = make_semantic4_mapping(semantic_path)

    group_of_label = np.asarray(
        freq3["group_of_label"], dtype=np.int8
    )

    # Generate a reproducible provisional frequency curve for mock labels.
    freqs = make_mock_frequencies(group_of_label)
    # The mapping is based on actual train counts when available, as specified.
    freq3 = make_freq3_mapping(train_counts, freqs, train_source)
    sem4 = make_semantic4_mapping(semantic_path)
    group_of_label = np.asarray(freq3["group_of_label"], dtype=np.int8)

    # Separate deterministic streams so changing one split does not perturb another.
    arrays_meta: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for split_index, split in enumerate(("val", "test")):
        rng = np.random.default_rng(SEED + split_index)
        P, Y = make_split(n, rng, freqs, group_of_label)
        for prefix, arr in (("P", P), ("Y", Y)):
            file_path = out_dir / f"{prefix}_{split}.npy"
            np.save(file_path, arr, allow_pickle=False)
            hashes[str(file_path.relative_to(SCRIPT_DIR))] = sha256(file_path)
            arrays_meta[f"{prefix}_{split}"] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    # Write each mapping with the same key schema expected by the project.
    for filename, payload in (
        ("mapping_freq3_mock.json", freq3),
        ("mapping_semantic4_mock.json", sem4),
    ):
        path = mappings_dir / filename
        write_json(path, payload)
        hashes[str(path.relative_to(SCRIPT_DIR))] = sha256(path)

    folds_rng = np.random.default_rng(SEED)
    _, fold_labels = make_split(2066, folds_rng, freqs, group_of_label)
    folds = make_cv_folds(fold_labels)
    folds_path = SCRIPT_DIR / "cv_folds_mock.npy"
    np.save(folds_path, folds, allow_pickle=False)
    hashes[str(folds_path.relative_to(SCRIPT_DIR))] = sha256(folds_path)

    golden_path = SCRIPT_DIR / "golden_cases.json"
    write_json(golden_path, golden_cases())
    hashes[str(golden_path.relative_to(SCRIPT_DIR))] = sha256(golden_path)

    meta = {
        "seed": SEED,
        "n": n,
        "n_labels": N_LABELS,
        "neutral_idx": neutral_idx,
        "parameters": {"a_head": A_HEAD, "a_tail": A_TAIL, "b": BIAS_B, "sigma": SIGMA},
        "train_mapping_source": train_source,
        "arrays": arrays_meta,
        "sha256": hashes,
        "reproducibility_note": "meta.json does not hash itself; all other generated files are hashed.",
    }
    meta_path = out_dir / "meta.json"
    write_json(meta_path, meta)
    # Record the metadata file's own hash in a sidecar-free top-level manifest.
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-mapping", type=Path, default=DEFAULT_SEMANTIC_MAPPING)
    parser.add_argument(
        "--allow-mock-freq3", action="store_true",
        help="Allow frequency mapping from the mock frequency curve if train CSV is absent.",
    )
    args = parser.parse_args()

    neutral_idx = load_constants_neutral_idx()
    if neutral_idx != NEUTRAL_IDX:
        raise ValueError(
            f"NEUTRAL_IDX trong constants.py là {neutral_idx}, "
            f"nhưng danh sách nhãn hiện khai báo neutral ở {NEUTRAL_IDX}."
        )

    train_counts: np.ndarray | None
    train_source: str
    try:
        train_counts, train_source = load_train_counts(DEFAULT_TRAIN_CSV)
    except FileNotFoundError:
        if not args.allow_mock_freq3:
            raise
        train_counts = None
        train_source = "mock frequency curve; --allow-mock-freq3"
        warnings.warn(
            "Đang tạo mapping freq3 từ tần suất mock, không phải train thật.",
            RuntimeWarning,
        )

    # One global deterministic seed controls all mock artifact generation.
    build("mock_small", 200, train_counts, train_source, args.semantic_mapping, neutral_idx)
    build("mock_fullshape", 2066, train_counts, train_source, args.semantic_mapping, neutral_idx)
    print(f"Output: {SCRIPT_DIR}")
    print("Đã tạo: mock_small/, mock_fullshape/, cv_folds_mock.npy, golden_cases.json")

if __name__ == "__main__":
    main()