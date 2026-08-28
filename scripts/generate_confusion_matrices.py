"""Generate confusion matrix outputs for current GAMEEMO baseline experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import (
    BaselineFeatureDataset,
    build_bandpower_feature_dataset,
    build_log_relative_bandpower_feature_dataset,
    build_stat_feature_dataset,
)
from src.config import LABEL_MAP, RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import make_loso_splits, make_subject_dependent_split, validate_loso_split
from src.windowing import DEFAULT_WINDOW_SAMPLES


RANDOM_SEED = 0
TEST_SIZE = 0.2
LABEL_ORDER = list(LABEL_MAP)
CONFUSION_OUTPUT = RESULTS_DIR / "baseline_confusion_matrices.csv"
NOTES_OUTPUT = RESULTS_DIR / "baseline_confusion_notes.md"


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


def dataset_builders():
    return [
        ("statistical", build_stat_feature_dataset),
        ("bandpower", build_bandpower_feature_dataset),
        ("log_relative_bandpower", build_log_relative_bandpower_feature_dataset),
    ]


def confusion_rows(
    *,
    protocol: str,
    feature_set: str,
    model_name: str,
    heldout_subject: str,
    y_true,
    y_pred,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = Counter(zip(y_true, y_pred))
    true_totals = Counter(y_true)
    for true_label in LABEL_ORDER:
        total = true_totals[true_label]
        for predicted_label in LABEL_ORDER:
            count = counts[(true_label, predicted_label)]
            normalized = count / total if total else 0.0
            rows.append(
                {
                    "protocol": protocol,
                    "feature_set": feature_set,
                    "model_name": model_name,
                    "heldout_subject": heldout_subject,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "normalized_by_true": f"{normalized:.6f}",
                }
            )
    return rows


def subject_dependent_confusion(dataset: BaselineFeatureDataset, feature_set: str) -> list[dict[str, object]]:
    import numpy as np

    metadata = dataset.metadata
    labels = np.asarray([item.label for item in metadata])
    split = make_subject_dependent_split(metadata, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=True)

    rows: list[dict[str, object]] = []
    for model_name in model_names():
        model = make_model(model_name)
        model.fit(dataset.features[split.train_indices], labels[split.train_indices])
        y_pred = model.predict(dataset.features[split.test_indices])
        rows.extend(
            confusion_rows(
                protocol="subject_dependent",
                feature_set=feature_set,
                model_name=model_name,
                heldout_subject="ALL",
                y_true=labels[split.test_indices],
                y_pred=y_pred,
            )
        )
        print(f"subject_dependent {feature_set} {model_name}: confusion rows added")
    return rows


def loso_confusion(dataset: BaselineFeatureDataset, feature_set: str) -> list[dict[str, object]]:
    import numpy as np

    metadata = dataset.metadata
    labels = np.asarray([item.label for item in metadata])
    splits = make_loso_splits(metadata)

    rows: list[dict[str, object]] = []
    for model_name in model_names():
        all_true = []
        all_pred = []
        for split in splits:
            validate_loso_split(metadata, split)
            assert_no_subject_leakage(metadata, split.train_indices, split.test_indices)
            model = make_model(model_name)
            model.fit(dataset.features[split.train_indices], labels[split.train_indices])
            y_pred = model.predict(dataset.features[split.test_indices])
            all_true.extend(labels[split.test_indices])
            all_pred.extend(y_pred)

        rows.extend(
            confusion_rows(
                protocol="loso",
                feature_set=feature_set,
                model_name=model_name,
                heldout_subject="ALL",
                y_true=all_true,
                y_pred=all_pred,
            )
        )
        print(f"loso {feature_set} {model_name}: aggregate confusion rows added")
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_off_diagonal_confusions(rows: list[dict[str, object]], n: int = 8) -> list[dict[str, object]]:
    return sorted(
        [
            row
            for row in rows
            if row["heldout_subject"] == "ALL" and row["true_label"] != row["predicted_label"]
        ],
        key=lambda row: int(row["count"]),
        reverse=True,
    )[:n]


def write_notes(output_path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Baseline Confusion Matrix Notes",
        "",
        "This file summarizes aggregate confusion matrices for the current GAMEEMO baseline settings.",
        "",
        "Rows are saved in `results/baseline_confusion_matrices.csv`; each block is normalized by true label.",
        "",
        "## Largest Aggregate Confusions",
        "",
    ]
    for row in top_off_diagonal_confusions(rows):
        lines.append(
            f"- {row['protocol']} / {row['feature_set']} / {row['model_name']}: "
            f"{row['true_label']} -> {row['predicted_label']} "
            f"count={row['count']} normalized={row['normalized_by_true']}"
        )

    lines.extend(
        [
            "",
            "## Use In Meeting",
            "",
            "- Use this to show which emotion conditions are most often confused, not just the headline accuracy.",
            "- The LOSO blocks are especially useful because they aggregate predictions from subject-held-out folds.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confusion matrix CSVs for GAMEEMO baselines.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=CONFUSION_OUTPUT)
    parser.add_argument("--notes-output", type=Path, default=NOTES_OUTPUT)
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debugging limit on the number of recordings to load.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for feature_set, builder in dataset_builders():
        dataset = builder(args.root, limit_records=args.limit_records)
        print(f"{feature_set}: records={dataset.n_records} features={dataset.features.shape}")
        rows.extend(subject_dependent_confusion(dataset, feature_set))
        rows.extend(loso_confusion(dataset, feature_set))

    write_csv(args.output, rows)
    write_notes(args.notes_output, rows)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.notes_output}")


if __name__ == "__main__":
    main()
