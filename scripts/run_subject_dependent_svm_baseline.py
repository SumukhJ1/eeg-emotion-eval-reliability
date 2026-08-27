"""Run a subject-dependent GAMEEMO linear SVM baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_stat_feature_dataset
from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import make_subject_dependent_split
from src.windowing import DEFAULT_WINDOW_SAMPLES


RANDOM_SEED = 0
TEST_SIZE = 0.2
MODEL_NAME = "StandardScaler+LinearSVC"
RESULTS_PATH = RESULTS_DIR / "subject_dependent_svm_baseline.csv"


def write_result_csv(
    output_path: Path,
    *,
    n_records: int,
    n_windows: int,
    n_features: int,
    train_windows: int,
    test_windows: int,
    stratified: bool,
    accuracy: float,
    macro_f1: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model_name",
        "random_seed",
        "test_size",
        "window_samples",
        "n_records",
        "n_windows",
        "n_features",
        "train_windows",
        "test_windows",
        "stratified",
        "accuracy",
        "macro_f1",
    ]
    row = {
        "experiment": "subject_dependent_linear_svm_stat_features",
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "window_samples": DEFAULT_WINDOW_SAMPLES,
        "n_records": n_records,
        "n_windows": n_windows,
        "n_features": n_features,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "stratified": stratified,
        "accuracy": f"{accuracy:.6f}",
        "macro_f1": f"{macro_f1:.6f}",
    }

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAMEEMO subject-dependent linear SVM baseline.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Path to write the result CSV.")
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debugging limit on the number of recordings to load.",
    )
    args = parser.parse_args()

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    dataset = build_stat_feature_dataset(args.root, limit_records=args.limit_records)
    features = dataset.features
    metadata = dataset.metadata
    feature_names = dataset.feature_names
    n_records = dataset.n_records
    labels = np.asarray([item.label for item in metadata])

    split = make_subject_dependent_split(metadata, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=True)
    x_train = features[split.train_indices]
    x_test = features[split.test_indices]
    y_train = labels[split.train_indices]
    y_test = labels[split.test_indices]

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(random_state=RANDOM_SEED, max_iter=50000, dual=False),
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    write_result_csv(
        args.output,
        n_records=n_records,
        n_windows=len(metadata),
        n_features=len(feature_names),
        train_windows=len(split.train_indices),
        test_windows=len(split.test_indices),
        stratified=split.stratified,
        accuracy=accuracy,
        macro_f1=macro_f1,
    )

    print(f"model_name: {MODEL_NAME}")
    print(f"records: {n_records}")
    print(f"features: {features.shape}")
    print(f"train_windows: {len(split.train_indices)}")
    print(f"test_windows: {len(split.test_indices)}")
    print(f"accuracy: {accuracy:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
