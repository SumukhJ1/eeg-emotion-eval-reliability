"""Audit subject-dependent splits matched to the LOSO window budget."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_window_dataset
from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import make_loso_splits, make_subject_dependent_matched_budget_split


DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
VAL_SIZE = 0.2
WINDOW_SAMPLES = 512
AUDIT_PATH = RESULTS_DIR / "subject_dependent_matched_budget_split_audit.csv"
SUMMARY_PATH = RESULTS_DIR / "subject_dependent_matched_budget_split_summary.csv"


def split_train_validation(train_indices: list[int], labels, val_size: float, random_seed: int) -> tuple[list[int], list[int]]:
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


def count_field(metadata, indices: list[int], field: str) -> Counter[str]:
    return Counter(str(getattr(metadata[idx], field)) for idx in indices)


def append_count_rows(
    rows: list[dict[str, object]],
    *,
    protocol: str,
    random_seed: int,
    split_name: str,
    field: str,
    counts: Counter[str],
    total_windows: int,
) -> None:
    for value in sorted(counts):
        rows.append(
            {
                "protocol": protocol,
                "random_seed": random_seed,
                "split": split_name,
                "field": field,
                "value": value,
                "count": counts[value],
                "proportion": f"{counts[value] / total_windows:.6f}",
                "split_total_windows": total_windows,
            }
        )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in raw.split(",") if seed.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and audit a matched-budget subject-dependent GAMEEMO split.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT)
    parser.add_argument("--window-samples", type=int, default=WINDOW_SAMPLES)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    dataset = build_window_dataset(args.root, window_samples=args.window_samples)
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        loso_splits = make_loso_splits(dataset.metadata)
        loso_fit_sizes: list[int] = []
        loso_val_sizes: list[int] = []
        loso_test_sizes: list[int] = []
        for split in loso_splits:
            fit_indices, val_indices = split_train_validation(
                split.train_indices,
                dataset.labels,
                val_size=args.val_size,
                random_seed=seed,
            )
            loso_fit_sizes.append(len(fit_indices))
            loso_val_sizes.append(len(val_indices))
            loso_test_sizes.append(len(split.test_indices))

        fit_windows = round(statistics.mean(loso_fit_sizes))
        val_windows = round(statistics.mean(loso_val_sizes))
        test_windows = round(statistics.mean(loso_test_sizes))
        matched_split = make_subject_dependent_matched_budget_split(
            dataset.metadata,
            fit_windows=fit_windows,
            val_windows=val_windows,
            test_windows=test_windows,
            random_state=seed,
            stratify=True,
        )

        for split_name, indices in [
            ("fit", matched_split.fit_indices),
            ("validation", matched_split.val_indices),
            ("test", matched_split.test_indices),
        ]:
            for field in ["label", "game", "subject"]:
                append_count_rows(
                    audit_rows,
                    protocol=matched_split.split_name,
                    random_seed=seed,
                    split_name=split_name,
                    field=field,
                    counts=count_field(dataset.metadata, indices, field),
                    total_windows=len(indices),
                )

        fit_keys = {(dataset.metadata[idx].subject, int(idx)) for idx in matched_split.fit_indices}
        val_keys = {(dataset.metadata[idx].subject, int(idx)) for idx in matched_split.val_indices}
        test_keys = {(dataset.metadata[idx].subject, int(idx)) for idx in matched_split.test_indices}
        train_test_window_overlap = len(fit_keys & test_keys)
        validation_test_window_overlap = len(val_keys & test_keys)

        summary_rows.extend(
            [
                {
                    "protocol": "loso_mean_budget",
                    "window_samples": args.window_samples,
                    "random_seed": seed,
                    "n_records": dataset.n_records,
                    "n_windows": len(dataset.metadata),
                    "fit_windows": fit_windows,
                    "validation_windows": val_windows,
                    "test_windows": test_windows,
                    "n_folds": len(loso_splits),
                    "stratified": "",
                    "train_test_window_overlap": "",
                    "validation_test_window_overlap": "",
                    "leakage_failure": "",
                },
                {
                    "protocol": matched_split.split_name,
                    "window_samples": args.window_samples,
                    "random_seed": seed,
                    "n_records": dataset.n_records,
                    "n_windows": len(dataset.metadata),
                    "fit_windows": len(matched_split.fit_indices),
                    "validation_windows": len(matched_split.val_indices),
                    "test_windows": len(matched_split.test_indices),
                    "n_folds": 1,
                    "stratified": matched_split.stratified,
                    "train_test_window_overlap": train_test_window_overlap,
                    "validation_test_window_overlap": validation_test_window_overlap,
                    "leakage_failure": train_test_window_overlap > 0 or validation_test_window_overlap > 0,
                },
            ]
        )

    write_csv(
        args.audit_output,
        audit_rows,
        ["protocol", "random_seed", "split", "field", "value", "count", "proportion", "split_total_windows"],
    )
    write_csv(
        args.summary_output,
        summary_rows,
        [
            "protocol",
            "window_samples",
            "random_seed",
            "n_records",
            "n_windows",
            "fit_windows",
            "validation_windows",
            "test_windows",
            "n_folds",
            "stratified",
            "train_test_window_overlap",
            "validation_test_window_overlap",
            "leakage_failure",
        ],
    )

    failures = [row for row in summary_rows if row["leakage_failure"] is True]
    print(f"windows: {len(dataset.metadata)}")
    print(f"seeds: {','.join(str(seed) for seed in args.seeds)}")
    print(f"matched_splits_audited: {len(args.seeds)}")
    print(f"window_leakage_failures: {len(failures)}")
    print(f"wrote: {args.audit_output}")
    print(f"wrote: {args.summary_output}")
    if failures:
        raise SystemExit("Matched-budget split leakage audit failed")


if __name__ == "__main__":
    main()
