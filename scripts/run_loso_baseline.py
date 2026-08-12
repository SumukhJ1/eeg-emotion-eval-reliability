"""Run a first leave-one-subject-out GAMEEMO logistic regression baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.features import extract_statistical_features
from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.splits import make_loso_splits, validate_loso_split
from src.windowing import DEFAULT_WINDOW_SAMPLES, window_gameemo_record


RANDOM_SEED = 0
LOSO_RESULTS_PATH = RESULTS_DIR / "loso_baseline.csv"
LOSO_SUMMARY_PATH = RESULTS_DIR / "loso_summary.csv"


def build_feature_dataset(root: Path, limit_records: int | None = None):
    import numpy as np

    records = discover_records(root, limit=limit_records)
    if not records:
        raise ValueError(f"No GAMEEMO preprocessed CSV records found under {root}")

    feature_blocks = []
    metadata = []
    feature_names: list[str] | None = None

    for record in records:
        loaded_record, data = load_record(record)
        windows, window_metadata = window_gameemo_record(loaded_record, data, window_samples=DEFAULT_WINDOW_SAMPLES)
        features, names = extract_statistical_features(windows, channels=loaded_record.channels)

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Feature names changed for {loaded_record.source_file}")

        feature_blocks.append(features)
        metadata.extend(window_metadata)

    return np.vstack(feature_blocks), metadata, feature_names or [], len(records)


def assert_no_subject_leakage(metadata, train_indices: list[int], test_indices: list[int]) -> None:
    train_subjects = {metadata[idx].subject for idx in train_indices}
    test_subjects = {metadata[idx].subject for idx in test_indices}
    leaked_subjects = train_subjects & test_subjects
    if leaked_subjects:
        raise ValueError(f"Subject leakage detected: {sorted(leaked_subjects)}")


def write_loso_results(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "train_windows",
        "test_windows",
        "accuracy",
        "macro_f1",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_loso_summary(
    output_path: Path,
    *,
    n_records: int,
    n_windows: int,
    n_features: int,
    n_folds: int,
    mean_accuracy: float,
    std_accuracy: float,
    mean_macro_f1: float,
    std_macro_f1: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "random_seed",
        "window_samples",
        "n_records",
        "n_windows",
        "n_features",
        "n_folds",
        "mean_accuracy",
        "std_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
    ]
    row = {
        "experiment": "loso_logistic_regression_stat_features",
        "random_seed": RANDOM_SEED,
        "window_samples": DEFAULT_WINDOW_SAMPLES,
        "n_records": n_records,
        "n_windows": n_windows,
        "n_features": n_features,
        "n_folds": n_folds,
        "mean_accuracy": f"{mean_accuracy:.6f}",
        "std_accuracy": f"{std_accuracy:.6f}",
        "mean_macro_f1": f"{mean_macro_f1:.6f}",
        "std_macro_f1": f"{std_macro_f1:.6f}",
    }
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAMEEMO LOSO logistic regression baseline.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=LOSO_RESULTS_PATH, help="Path to write per-subject results.")
    parser.add_argument("--summary-output", type=Path, default=LOSO_SUMMARY_PATH, help="Path to write summary results.")
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debugging limit on the number of recordings to load.",
    )
    args = parser.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, metadata, feature_names, n_records = build_feature_dataset(args.root, limit_records=args.limit_records)
    labels = np.asarray([item.label for item in metadata])
    loso_splits = make_loso_splits(metadata)

    rows: list[dict[str, object]] = []
    for split in loso_splits:
        validate_loso_split(metadata, split)
        assert_no_subject_leakage(metadata, split.train_indices, split.test_indices)

        x_train = features[split.train_indices]
        x_test = features[split.test_indices]
        y_train = labels[split.train_indices]
        y_test = labels[split.test_indices]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        rows.append(
            {
                "subject": split.test_subject,
                "train_windows": len(split.train_indices),
                "test_windows": len(split.test_indices),
                "accuracy": f"{accuracy:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
            }
        )
        print(
            f"{split.test_subject}: train_windows={len(split.train_indices)} "
            f"test_windows={len(split.test_indices)} accuracy={accuracy:.6f} macro_f1={macro_f1:.6f}"
        )

    accuracies = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
    macro_f1s = np.asarray([float(row["macro_f1"]) for row in rows], dtype=float)

    write_loso_results(args.output, rows)
    write_loso_summary(
        args.summary_output,
        n_records=n_records,
        n_windows=len(metadata),
        n_features=len(feature_names),
        n_folds=len(rows),
        mean_accuracy=float(accuracies.mean()),
        std_accuracy=float(accuracies.std()),
        mean_macro_f1=float(macro_f1s.mean()),
        std_macro_f1=float(macro_f1s.std()),
    )

    print(f"records: {n_records}")
    print(f"features: {features.shape}")
    print(f"folds: {len(rows)}")
    print(f"mean_accuracy: {accuracies.mean():.6f}")
    print(f"std_accuracy: {accuracies.std():.6f}")
    print(f"mean_macro_f1: {macro_f1s.mean():.6f}")
    print(f"std_macro_f1: {macro_f1s.std():.6f}")
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")


if __name__ == "__main__":
    main()
