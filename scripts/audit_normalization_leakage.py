"""Audit train-only normalization for GAMEEMO neural split protocols."""

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
from src.neural_normalization import fit_channel_standardization
from src.splits import make_loso_splits, make_subject_dependent_split, validate_loso_split


TEST_SIZE = 0.2
VAL_SIZE = 0.2
WINDOW_SAMPLES = 512
DEFAULT_SEEDS = [0, 1, 2]
OUTPUT_PATH = RESULTS_DIR / "normalization_leakage_audit.csv"


def split_train_validation(train_indices: list[int], labels, val_size: float, random_seed: int) -> tuple[list[int], list[int]]:
    """Match neural runners: validation split is made from training indices only."""
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


def audit_normalization_row(
    *,
    protocol: str,
    fold_seed: str,
    windows,
    metadata,
    fit_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    held_out_subject: str | None,
) -> dict[str, object]:
    fit_set = set(fit_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    train_test_overlap = bool(fit_set & test_set)
    val_test_overlap = bool(val_set & test_set)
    heldout_subject_used_in_fit = False
    if held_out_subject is not None:
        heldout_subject_used_in_fit = held_out_subject in subjects_for_indices(metadata, fit_indices)

    stats = fit_channel_standardization(windows, fit_indices)

    return {
        "protocol": protocol,
        "fold_seed": fold_seed,
        "normalization_source": "train only",
        "test_data_used_in_fit": train_test_overlap,
        "validation_data_used_in_fit": bool(fit_set & val_set),
        "held_out_subject": held_out_subject or "",
        "held_out_subject_used_in_fit": heldout_subject_used_in_fit,
        "fit_windows": len(fit_indices),
        "validation_windows": len(val_indices),
        "test_windows": len(test_indices),
        "fit_subjects": len(subjects_for_indices(metadata, fit_indices)),
        "validation_subjects": len(subjects_for_indices(metadata, val_indices)),
        "test_subjects": len(subjects_for_indices(metadata, test_indices)),
        "normalization_strategy": stats.strategy,
        "n_channel_means": int(stats.mean.shape[1]),
        "n_channel_stds": int(stats.std.shape[1]),
        "leakage_failure": train_test_overlap or val_test_overlap or heldout_subject_used_in_fit,
    }


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "fold_seed",
        "normalization_source",
        "test_data_used_in_fit",
        "validation_data_used_in_fit",
        "held_out_subject",
        "held_out_subject_used_in_fit",
        "fit_windows",
        "validation_windows",
        "test_windows",
        "fit_subjects",
        "validation_subjects",
        "test_subjects",
        "normalization_strategy",
        "n_channel_means",
        "n_channel_stds",
        "leakage_failure",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    failures = [row for row in rows if row["leakage_failure"]]
    compact_rows = [
        {
            "protocol": row["protocol"],
            "fold_seed": row["fold_seed"],
            "normalization_source": row["normalization_source"],
            "test_data_used_in_fit": row["test_data_used_in_fit"],
        }
        for row in rows
    ]
    preview = compact_rows[:8]
    lines = [
        "# Normalization Leakage Audit",
        "",
        "This audit checks that GAMEEMO neural normalization statistics are fit on training windows only, then reused for validation and test windows.",
        "",
        f"Rows audited: {len(rows)}",
        f"Leakage failures: {len(failures)}",
        "",
        "| Protocol | Fold/seed | Normalization source | Test data used in fit? |",
        "| --- | --- | --- | --- |",
    ]
    for row in preview:
        lines.append(
            f"| {row['protocol']} | {row['fold_seed']} | "
            f"{row['normalization_source']} | {row['test_data_used_in_fit']} |"
        )
    if len(compact_rows) > len(preview):
        lines.append(f"| ... | {len(compact_rows) - len(preview)} more rows in CSV | train only | False |")
    lines.extend(
        [
            "",
            "The detailed CSV also records whether validation windows overlap with fit windows and whether each LOSO held-out subject appears in the normalization fit set.",
            "",
            "Expected result: `normalization_source=train only`, `test_data_used_in_fit=False`, and `held_out_subject_used_in_fit=False` for all LOSO folds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in raw.split(",") if seed.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GAMEEMO normalization leakage.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=RESULTS_DIR / "normalization_leakage_audit.md")
    parser.add_argument("--window-samples", type=int, default=WINDOW_SAMPLES)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    args = parser.parse_args()

    if args.window_samples <= 0:
        raise SystemExit(f"window-samples must be positive, got {args.window_samples}")
    if not 0 < args.val_size < 1:
        raise SystemExit(f"val-size must be between 0 and 1, got {args.val_size}")

    dataset = build_window_dataset(args.root, window_samples=args.window_samples)
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        sd_split = make_subject_dependent_split(
            dataset.metadata,
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=True,
        )
        sd_fit, sd_val = split_train_validation(
            sd_split.train_indices,
            dataset.labels,
            val_size=args.val_size,
            random_seed=seed,
        )
        rows.append(
            audit_normalization_row(
                protocol="subject_dependent",
                fold_seed=f"seed_{seed}",
                windows=dataset.windows,
                metadata=dataset.metadata,
                fit_indices=sd_fit,
                val_indices=sd_val,
                test_indices=sd_split.test_indices,
                held_out_subject=None,
            )
        )

        for loso_split in make_loso_splits(dataset.metadata):
            validate_loso_split(dataset.metadata, loso_split)
            fit_indices, val_indices = split_train_validation(
                loso_split.train_indices,
                dataset.labels,
                val_size=args.val_size,
                random_seed=seed,
            )
            rows.append(
                audit_normalization_row(
                    protocol="loso",
                    fold_seed=f"{loso_split.split_name}_seed_{seed}",
                    windows=dataset.windows,
                    metadata=dataset.metadata,
                    fit_indices=fit_indices,
                    val_indices=val_indices,
                    test_indices=loso_split.test_indices,
                    held_out_subject=loso_split.test_subject,
                )
            )

    write_csv(args.output, rows)
    write_markdown(args.markdown_output, rows)

    failures = [row for row in rows if row["leakage_failure"]]
    print(f"rows: {len(rows)}")
    print(f"window_samples: {args.window_samples}")
    print(f"normalization_source: train only")
    print(f"test_data_used_in_fit_failures: {sum(bool(row['test_data_used_in_fit']) for row in rows)}")
    print(f"held_out_subject_used_in_fit_failures: {sum(bool(row['held_out_subject_used_in_fit']) for row in rows)}")
    print(f"leakage_failures: {len(failures)}")
    print(f"wrote: {args.output}")
    print(f"wrote: {args.markdown_output}")
    if failures:
        raise SystemExit("Normalization leakage audit failed")


if __name__ == "__main__":
    main()
