"""Run class-balanced GAMEEMO baselines with log-relative bandpower features."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import BaselineFeatureDataset, build_log_relative_bandpower_feature_dataset
from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import make_loso_splits, make_subject_dependent_split, validate_loso_split
from src.windowing import DEFAULT_WINDOW_SAMPLES


RANDOM_SEED = 0
TEST_SIZE = 0.2
FEATURE_SET = "log_relative_bandpower"
CLASS_WEIGHT = "balanced"
SUBJECT_DEPENDENT_OUTPUT = RESULTS_DIR / "subject_dependent_class_balanced_baselines.csv"
LOSO_OUTPUT = RESULTS_DIR / "loso_class_balanced_baselines.csv"
LOSO_SUMMARY_OUTPUT = RESULTS_DIR / "loso_class_balanced_summary.csv"


def assert_no_subject_leakage(metadata, train_indices: list[int], test_indices: list[int]) -> None:
    train_subjects = {metadata[idx].subject for idx in train_indices}
    test_subjects = {metadata[idx].subject for idx in test_indices}
    leaked_subjects = train_subjects & test_subjects
    if leaked_subjects:
        raise ValueError(f"Subject leakage detected: {sorted(leaked_subjects)}")


def make_model(model_name: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if model_name == "LogisticRegression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight=CLASS_WEIGHT,
            ),
        )
    if model_name == "LinearSVM":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                random_state=RANDOM_SEED,
                max_iter=50000,
                dual=False,
                class_weight=CLASS_WEIGHT,
            ),
        )
    raise ValueError(f"Unknown model name: {model_name}")


def model_names() -> list[str]:
    return ["LogisticRegression", "LinearSVM"]


def evaluate_model(model, x_train, y_train, x_test, y_test) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return float(accuracy), float(macro_f1)


def run_subject_dependent(dataset: BaselineFeatureDataset) -> list[dict[str, object]]:
    import numpy as np

    metadata = dataset.metadata
    labels = np.asarray([item.label for item in metadata])
    split = make_subject_dependent_split(metadata, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=True)

    rows: list[dict[str, object]] = []
    for model_name in model_names():
        model = make_model(model_name)
        accuracy, macro_f1 = evaluate_model(
            model,
            dataset.features[split.train_indices],
            labels[split.train_indices],
            dataset.features[split.test_indices],
            labels[split.test_indices],
        )
        rows.append(
            {
                "protocol": "subject_dependent",
                "model_name": model_name,
                "feature_set": FEATURE_SET,
                "class_weight": CLASS_WEIGHT,
                "random_seed": RANDOM_SEED,
                "test_size": TEST_SIZE,
                "window_samples": DEFAULT_WINDOW_SAMPLES,
                "n_records": dataset.n_records,
                "n_windows": len(metadata),
                "n_features": len(dataset.feature_names),
                "train_windows": len(split.train_indices),
                "test_windows": len(split.test_indices),
                "stratified": split.stratified,
                "accuracy": f"{accuracy:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
            }
        )
        print(
            f"subject_dependent {model_name} class_weight={CLASS_WEIGHT}: "
            f"accuracy={accuracy:.6f} macro_f1={macro_f1:.6f}"
        )

    return rows


def run_loso(dataset: BaselineFeatureDataset) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import numpy as np

    metadata = dataset.metadata
    labels = np.asarray([item.label for item in metadata])
    splits = make_loso_splits(metadata)
    fold_rows: list[dict[str, object]] = []

    for model_name in model_names():
        for split in splits:
            validate_loso_split(metadata, split)
            assert_no_subject_leakage(metadata, split.train_indices, split.test_indices)
            model = make_model(model_name)
            accuracy, macro_f1 = evaluate_model(
                model,
                dataset.features[split.train_indices],
                labels[split.train_indices],
                dataset.features[split.test_indices],
                labels[split.test_indices],
            )
            row = {
                "model_name": model_name,
                "subject": split.test_subject,
                "feature_set": FEATURE_SET,
                "class_weight": CLASS_WEIGHT,
                "train_windows": len(split.train_indices),
                "test_windows": len(split.test_indices),
                "accuracy": f"{accuracy:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
            }
            fold_rows.append(row)
            print(
                f"loso {model_name} {split.test_subject} class_weight={CLASS_WEIGHT}: "
                f"accuracy={accuracy:.6f} macro_f1={macro_f1:.6f}"
            )

    summary_rows: list[dict[str, object]] = []
    for model_name in model_names():
        model_rows = [row for row in fold_rows if row["model_name"] == model_name]
        accuracies = np.asarray([float(row["accuracy"]) for row in model_rows], dtype=float)
        macro_f1s = np.asarray([float(row["macro_f1"]) for row in model_rows], dtype=float)
        summary_rows.append(
            {
                "protocol": "loso",
                "model_name": model_name,
                "feature_set": FEATURE_SET,
                "class_weight": CLASS_WEIGHT,
                "random_seed": RANDOM_SEED,
                "window_samples": DEFAULT_WINDOW_SAMPLES,
                "n_records": dataset.n_records,
                "n_windows": len(metadata),
                "n_features": len(dataset.feature_names),
                "n_folds": len(model_rows),
                "mean_accuracy": f"{accuracies.mean():.6f}",
                "std_accuracy": f"{accuracies.std():.6f}",
                "mean_macro_f1": f"{macro_f1s.mean():.6f}",
                "std_macro_f1": f"{macro_f1s.std():.6f}",
            }
        )
        print(
            f"loso {model_name} class_weight={CLASS_WEIGHT}: "
            f"mean_accuracy={accuracies.mean():.6f} mean_macro_f1={macro_f1s.mean():.6f}"
        )

    return fold_rows, summary_rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run class-balanced GAMEEMO log-relative bandpower baselines.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--subject-output", type=Path, default=SUBJECT_DEPENDENT_OUTPUT)
    parser.add_argument("--loso-output", type=Path, default=LOSO_OUTPUT)
    parser.add_argument("--loso-summary-output", type=Path, default=LOSO_SUMMARY_OUTPUT)
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debugging limit on the number of recordings to load.",
    )
    args = parser.parse_args()

    dataset = build_log_relative_bandpower_feature_dataset(args.root, limit_records=args.limit_records)
    print(f"records: {dataset.n_records}")
    print(f"features: {dataset.features.shape}")

    subject_rows = run_subject_dependent(dataset)
    loso_rows, loso_summary_rows = run_loso(dataset)

    write_csv(args.subject_output, subject_rows)
    write_csv(args.loso_output, loso_rows)
    write_csv(args.loso_summary_output, loso_summary_rows)

    print(f"wrote: {args.subject_output}")
    print(f"wrote: {args.loso_output}")
    print(f"wrote: {args.loso_summary_output}")


if __name__ == "__main__":
    main()
