"""Audit GAMEEMO LOSO splits for explicit subject leakage."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_window_dataset
from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import MetadataSplit, make_loso_splits, validate_loso_split


RANDOM_SEED = 0
VAL_SIZE = 0.2
WINDOW_SAMPLES = 512
OUTPUT_PATH = RESULTS_DIR / "loso_leakage_audit.csv"


def split_train_validation(train_indices: list[int], labels, val_size: float, random_seed: int) -> tuple[list[int], list[int]]:
    """Match the LOSO neural runners: stratify validation split when feasible."""
    from sklearn.model_selection import train_test_split

    train_labels = labels[train_indices]
    try:
        fit_indices, val_indices = train_test_split(
            train_indices,
            test_size=val_size,
            random_state=random_seed,
            stratify=train_labels,
        )
    except ValueError:
        fit_indices, val_indices = train_test_split(
            train_indices,
            test_size=val_size,
            random_state=random_seed,
            stratify=None,
        )
    return sorted(int(idx) for idx in fit_indices), sorted(int(idx) for idx in val_indices)


def subjects_for_indices(metadata, indices: list[int]) -> set[str]:
    return {metadata[idx].subject for idx in indices}


def audit_fold(metadata, labels, split: MetadataSplit, random_seed: int, val_size: float) -> dict[str, object]:
    validate_loso_split(metadata, split)
    heldout = split.test_subject
    if heldout is None:
        raise ValueError("LOSO split missing held-out subject")

    fit_indices, val_indices = split_train_validation(
        split.train_indices,
        labels,
        val_size=val_size,
        random_seed=random_seed,
    )

    train_subjects = subjects_for_indices(metadata, fit_indices)
    val_subjects = subjects_for_indices(metadata, val_indices)
    test_subjects = subjects_for_indices(metadata, split.test_indices)

    heldout_in_train = heldout in train_subjects
    heldout_in_val = heldout in val_subjects
    heldout_in_test_only = test_subjects == {heldout} and not heldout_in_train and not heldout_in_val

    return {
        "fold": split.split_name,
        "held_out_subject": heldout,
        "held_out_in_train": heldout_in_train,
        "held_out_in_val": heldout_in_val,
        "held_out_in_test_only": heldout_in_test_only,
        "train_subjects": ";".join(sorted(train_subjects)),
        "val_subjects": ";".join(sorted(val_subjects)),
        "test_subject": ";".join(sorted(test_subjects)),
        "train_window_count": len(fit_indices),
        "val_window_count": len(val_indices),
        "test_window_count": len(split.test_indices),
    }


def write_audit(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "held_out_subject",
        "held_out_in_train",
        "held_out_in_val",
        "held_out_in_test_only",
        "train_subjects",
        "val_subjects",
        "test_subject",
        "train_window_count",
        "val_window_count",
        "test_window_count",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GAMEEMO LOSO folds for held-out subject leakage.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--window-samples", type=int, default=WINDOW_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    args = parser.parse_args()

    if args.window_samples <= 0:
        raise SystemExit(f"window-samples must be positive, got {args.window_samples}")
    if not 0 < args.val_size < 1:
        raise SystemExit(f"val-size must be between 0 and 1, got {args.val_size}")

    dataset = build_window_dataset(args.root, window_samples=args.window_samples)
    rows = [
        audit_fold(dataset.metadata, dataset.labels, split, random_seed=args.random_seed, val_size=args.val_size)
        for split in make_loso_splits(dataset.metadata)
    ]
    write_audit(args.output, rows)

    leaked_rows = [
        row
        for row in rows
        if row["held_out_in_train"] or row["held_out_in_val"] or not row["held_out_in_test_only"]
    ]
    print(f"folds: {len(rows)}")
    print(f"window_samples: {args.window_samples}")
    print(f"leakage_failures: {len(leaked_rows)}")
    print(f"all_held_out_subjects_test_only: {not leaked_rows}")
    print(f"wrote: {args.output}")
    if leaked_rows:
        raise SystemExit("LOSO leakage audit failed")


if __name__ == "__main__":
    main()
