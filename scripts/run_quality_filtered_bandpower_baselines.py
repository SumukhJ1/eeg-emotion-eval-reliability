"""Run GAMEEMO bandpower baselines after conservative quality filtering."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import BaselineFeatureDataset
from src.config import RESULTS_DIR, SAMPLING_RATE
from src.features import EEG_BANDS, extract_bandpower_features
from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.quality_filter import DEFAULT_VARIANCE_EPSILON, QualityFilterReport, conservative_quality_mask
from src.splits import make_loso_splits, make_subject_dependent_split, validate_loso_split
from src.windowing import DEFAULT_WINDOW_SAMPLES, EegWindowMetadata, window_gameemo_record


RANDOM_SEED = 0
TEST_SIZE = 0.2
FEATURE_SET = "fft_bandpower_quality_filtered"
SUBJECT_DEPENDENT_OUTPUT = RESULTS_DIR / "subject_dependent_bandpower_quality_filtered_baselines.csv"
LOSO_OUTPUT = RESULTS_DIR / "loso_bandpower_quality_filtered_baselines.csv"
LOSO_SUMMARY_OUTPUT = RESULTS_DIR / "loso_bandpower_quality_filtered_summary.csv"
FILTER_SUMMARY_OUTPUT = RESULTS_DIR / "bandpower_quality_filter_summary.csv"


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
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        )
    if model_name == "LinearSVM":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(random_state=RANDOM_SEED, max_iter=50000, dual=False),
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


def combine_reports(reports: list[QualityFilterReport]) -> QualityFilterReport:
    return QualityFilterReport(
        n_windows=sum(report.n_windows for report in reports),
        kept_windows=sum(report.kept_windows for report in reports),
        removed_windows=sum(report.removed_windows for report in reports),
        missing_value_windows=sum(report.missing_value_windows for report in reports),
        nonfinite_value_windows=sum(report.nonfinite_value_windows for report in reports),
        near_zero_variance_windows=sum(report.near_zero_variance_windows for report in reports),
        flat_channel_windows=sum(report.flat_channel_windows for report in reports),
        flat_channel_instances=sum(report.flat_channel_instances for report in reports),
        variance_epsilon=reports[0].variance_epsilon if reports else DEFAULT_VARIANCE_EPSILON,
    )


def build_filtered_bandpower_dataset(
    root: Path,
    limit_records: int | None,
    window_samples: int,
    sampling_rate: int,
    variance_epsilon: float,
) -> tuple[BaselineFeatureDataset, QualityFilterReport]:
    import numpy as np

    records = discover_records(root, limit=limit_records)
    if not records:
        raise ValueError(f"No GAMEEMO preprocessed CSV records found under {root}")

    feature_blocks = []
    metadata: list[EegWindowMetadata] = []
    reports: list[QualityFilterReport] = []
    feature_names: list[str] | None = None

    for record in records:
        loaded_record, data = load_record(record)
        windows, window_metadata = window_gameemo_record(
            loaded_record,
            data,
            window_samples=window_samples,
        )
        keep_mask, report = conservative_quality_mask(windows, variance_epsilon=variance_epsilon)
        reports.append(report)
        filtered_windows = windows[keep_mask]
        filtered_metadata = [item for item, keep in zip(window_metadata, keep_mask) if keep]
        if filtered_windows.shape[0] == 0:
            continue

        features, names = extract_bandpower_features(
            filtered_windows,
            channels=loaded_record.channels,
            sampling_rate=sampling_rate,
            bands=EEG_BANDS,
            mode="absolute",
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Feature names changed for {loaded_record.source_file}")

        feature_blocks.append(features)
        metadata.extend(filtered_metadata)

    if not feature_blocks:
        raise ValueError("Quality filtering removed all windows.")

    return (
        BaselineFeatureDataset(
            features=np.vstack(feature_blocks),
            metadata=metadata,
            feature_names=feature_names or [],
            n_records=len(records),
        ),
        combine_reports(reports),
    )


def run_subject_dependent(dataset: BaselineFeatureDataset, filter_report: QualityFilterReport) -> list[dict[str, object]]:
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
                "random_seed": RANDOM_SEED,
                "test_size": TEST_SIZE,
                "window_samples": DEFAULT_WINDOW_SAMPLES,
                "n_records": dataset.n_records,
                "n_windows_before_filter": filter_report.n_windows,
                "n_windows": len(metadata),
                "removed_windows": filter_report.removed_windows,
                "n_features": len(dataset.feature_names),
                "train_windows": len(split.train_indices),
                "test_windows": len(split.test_indices),
                "stratified": split.stratified,
                "accuracy": f"{accuracy:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
            }
        )
        print(f"subject_dependent {model_name}: accuracy={accuracy:.6f} macro_f1={macro_f1:.6f}")

    return rows


def run_loso(
    dataset: BaselineFeatureDataset,
    filter_report: QualityFilterReport,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
                "train_windows": len(split.train_indices),
                "test_windows": len(split.test_indices),
                "accuracy": f"{accuracy:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
            }
            fold_rows.append(row)
            print(f"loso {model_name} {split.test_subject}: accuracy={accuracy:.6f} macro_f1={macro_f1:.6f}")

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
                "random_seed": RANDOM_SEED,
                "window_samples": DEFAULT_WINDOW_SAMPLES,
                "n_records": dataset.n_records,
                "n_windows_before_filter": filter_report.n_windows,
                "n_windows": len(metadata),
                "removed_windows": filter_report.removed_windows,
                "n_features": len(dataset.feature_names),
                "n_folds": len(model_rows),
                "mean_accuracy": f"{accuracies.mean():.6f}",
                "std_accuracy": f"{accuracies.std():.6f}",
                "mean_macro_f1": f"{macro_f1s.mean():.6f}",
                "std_macro_f1": f"{macro_f1s.std():.6f}",
            }
        )
        print(
            f"loso {model_name}: mean_accuracy={accuracies.mean():.6f} "
            f"mean_macro_f1={macro_f1s.mean():.6f}"
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
    parser = argparse.ArgumentParser(description="Run quality-filtered GAMEEMO bandpower LR/SVM baselines.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--subject-output", type=Path, default=SUBJECT_DEPENDENT_OUTPUT)
    parser.add_argument("--loso-output", type=Path, default=LOSO_OUTPUT)
    parser.add_argument("--loso-summary-output", type=Path, default=LOSO_SUMMARY_OUTPUT)
    parser.add_argument("--filter-summary-output", type=Path, default=FILTER_SUMMARY_OUTPUT)
    parser.add_argument("--variance-epsilon", type=float, default=DEFAULT_VARIANCE_EPSILON)
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debugging limit on the number of recordings to load.",
    )
    args = parser.parse_args()

    dataset, filter_report = build_filtered_bandpower_dataset(
        root=args.root,
        limit_records=args.limit_records,
        window_samples=DEFAULT_WINDOW_SAMPLES,
        sampling_rate=SAMPLING_RATE,
        variance_epsilon=args.variance_epsilon,
    )
    print(f"records: {dataset.n_records}")
    print(f"features: {dataset.features.shape}")
    print(f"windows_before_filter: {filter_report.n_windows}")
    print(f"windows_after_filter: {filter_report.kept_windows}")
    print(f"removed_windows: {filter_report.removed_windows}")

    subject_rows = run_subject_dependent(dataset, filter_report)
    loso_rows, loso_summary_rows = run_loso(dataset, filter_report)
    filter_summary = {
        **asdict(filter_report),
        "filter_policy": "remove_missing_nonfinite_near_zero_variance_flat_channel_windows",
        "extreme_amplitude_policy": "flag_only",
        "extreme_variance_policy": "flag_only",
    }

    write_csv(args.subject_output, subject_rows)
    write_csv(args.loso_output, loso_rows)
    write_csv(args.loso_summary_output, loso_summary_rows)
    write_csv(args.filter_summary_output, [filter_summary])

    print(f"wrote: {args.subject_output}")
    print(f"wrote: {args.loso_output}")
    print(f"wrote: {args.loso_summary_output}")
    print(f"wrote: {args.filter_summary_output}")


if __name__ == "__main__":
    main()
