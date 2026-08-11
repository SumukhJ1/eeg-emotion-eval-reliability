"""Print demo subject-dependent and LOSO split counts for GAMEEMO windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gameemo_loader import GAMEEMO_ROOT, discover_records
from src.splits import label_counts, make_loso_splits, make_subject_dependent_split, subject_counts
from src.windowing import DEFAULT_WINDOW_SAMPLES, build_window_metadata


def build_metadata_from_records(limit_subjects: int | None = None):
    records = discover_records(GAMEEMO_ROOT)
    if limit_subjects is not None:
        subjects = sorted({record.subject for record in records})[:limit_subjects]
        records = [record for record in records if record.subject in subjects]

    metadata = []
    for record in records:
        n_windows = record.n_samples // DEFAULT_WINDOW_SAMPLES
        metadata.extend(build_window_metadata(record, n_windows=n_windows))

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo GAMEEMO metadata split generation.")
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=4,
        help="Limit demo metadata to the first N subjects. Use 0 for all subjects.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Subject-dependent test fraction.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for subject-dependent split.")
    parser.add_argument("--loso-folds", type=int, default=3, help="Number of LOSO folds to print.")
    args = parser.parse_args()

    limit_subjects = None if args.limit_subjects == 0 else args.limit_subjects
    metadata = build_metadata_from_records(limit_subjects=limit_subjects)

    print(f"window_metadata_count: {len(metadata)}")
    print(f"subjects: {len(subject_counts(metadata, range(len(metadata))))}")
    print(f"labels: {dict(label_counts(metadata, range(len(metadata))))}")

    split = make_subject_dependent_split(
        metadata,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )
    print("subject_dependent_split:")
    print(f"  train_windows: {len(split.train_indices)}")
    print(f"  test_windows: {len(split.test_indices)}")
    print(f"  stratified: {split.stratified}")
    print(f"  train_labels: {dict(label_counts(metadata, split.train_indices))}")
    print(f"  test_labels: {dict(label_counts(metadata, split.test_indices))}")

    loso_splits = make_loso_splits(metadata)
    print("loso_splits:")
    for fold in loso_splits[: args.loso_folds]:
        test_subjects = dict(subject_counts(metadata, fold.test_indices))
        print(
            f"  {fold.split_name}: train_windows={len(fold.train_indices)} "
            f"test_windows={len(fold.test_indices)} test_subjects={test_subjects}"
        )


if __name__ == "__main__":
    main()
