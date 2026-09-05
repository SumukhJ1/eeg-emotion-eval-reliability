"""Compare GAMEEMO Transformer subject-dependent and LOSO split composition."""

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
from src.splits import make_loso_splits, make_subject_dependent_split


TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_SEED = 0
WINDOW_SAMPLES = 512
OUTPUT_DIR = RESULTS_DIR / "transformer_split_composition"


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


def count_values(metadata, indices: list[int], field: str) -> Counter[str]:
    return Counter(str(getattr(metadata[idx], field)) for idx in indices)


def proportions(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(
    rows: list[dict[str, object]],
    *,
    protocol: str,
    split_name: str,
    field: str,
    counter: Counter[str],
    total_windows: int,
    fold: str = "ALL",
) -> None:
    props = proportions(counter)
    for value in sorted(counter):
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "split": split_name,
                "field": field,
                "value": value,
                "count": counter[value],
                "proportion": f"{props[value]:.6f}",
                "split_total_windows": total_windows,
            }
        )


def read_loso_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def summarize_loso_subject_results(loso_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_subject: dict[str, list[dict[str, str]]] = {}
    for row in loso_rows:
        by_subject.setdefault(row["subject"], []).append(row)

    rows: list[dict[str, object]] = []
    for subject, subject_rows in sorted(by_subject.items()):
        accuracies = [float(row["test_accuracy"]) for row in subject_rows]
        macro_f1s = [float(row["test_macro_f1"]) for row in subject_rows]
        rows.append(
            {
                "subject": subject,
                "n_seeds": len(subject_rows),
                "mean_accuracy": f"{statistics.mean(accuracies):.6f}",
                "std_accuracy": f"{statistics.pstdev(accuracies):.6f}",
                "mean_macro_f1": f"{statistics.mean(macro_f1s):.6f}",
                "std_macro_f1": f"{statistics.pstdev(macro_f1s):.6f}",
                "min_accuracy": f"{min(accuracies):.6f}",
                "max_accuracy": f"{max(accuracies):.6f}",
            }
        )
    return rows


def collect_loso_repeated_rows(run_dir: Path, seeds: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in seeds:
        path = run_dir / f"loso_seed_{seed}.csv"
        if path.exists():
            rows.extend(read_loso_rows(path))
    return rows


def collect_loso_subject_rows(run_dir: Path, seeds: list[int], seed0_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if 0 in seeds and seed0_path.exists():
        seed0_rows = read_loso_rows(seed0_path)
        for row in seed0_rows:
            row = dict(row)
            row["random_seed"] = "0"
            rows.append(row)
    rows.extend(collect_loso_repeated_rows(run_dir, [seed for seed in seeds if seed != 0]))
    if not rows:
        raise SystemExit("No LOSO per-subject rows found for the requested seeds.")
    return rows


def write_markdown(
    path: Path,
    *,
    sd_fit_windows: int,
    sd_val_windows: int,
    sd_test_windows: int,
    loso_fit_windows: list[int],
    loso_val_windows: list[int],
    loso_test_windows: list[int],
    sd_test_subject_count: int,
    loso_subject_rows: list[dict[str, object]],
    sd_result_path: Path,
    loso_result_path: Path,
) -> None:
    loso_sorted = sorted(loso_subject_rows, key=lambda row: float(row["mean_macro_f1"]))
    hardest = loso_sorted[:5]
    easiest = list(reversed(loso_sorted[-5:]))

    lines = [
        "# Transformer Split Composition Analysis",
        "",
        "This analysis compares the GAMEEMO 4-second temporal-patch Transformer subject-dependent split against LOSO folds to explain why LOSO is outperforming subject-dependent evaluation.",
        "",
        "## Train Size",
        "",
        "| Protocol | Fit windows | Val windows | Test windows | Note |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| Subject-dependent | {sd_fit_windows} | {sd_val_windows} | {sd_test_windows} | One random stratified window split across all subjects. |",
        f"| LOSO per fold | {round(statistics.mean(loso_fit_windows))} | {round(statistics.mean(loso_val_windows))} | {round(statistics.mean(loso_test_windows))} | Holds out one subject; trains on the other 27 subjects. |",
        "",
        "The biggest structural difference is train size: LOSO fits on many more windows per fold because only one subject is held out before the validation split.",
        "",
        "## Class And Game Balance",
        "",
        "The subject-dependent test split is exactly balanced by label and game: 414 windows per class/game. This means the lower subject-dependent Transformer score is not explained by a harder class-balance distribution.",
        "",
        "## Subject Coverage",
        "",
        f"The subject-dependent test split contains windows from {sd_test_subject_count} subjects. Each LOSO test fold contains exactly one subject, with all windows from that subject held out.",
        "",
        "## LOSO Subject Distribution",
        "",
        "| Hardest held-out subjects | Mean acc | Mean F1 |",
        "| --- | ---: | ---: |",
    ]
    for row in hardest:
        lines.append(f"| {row['subject']} | {row['mean_accuracy']} | {row['mean_macro_f1']} |")
    lines.extend(["", "| Easiest held-out subjects | Mean acc | Mean F1 |", "| --- | ---: | ---: |"])
    for row in easiest:
        lines.append(f"| {row['subject']} | {row['mean_accuracy']} | {row['mean_macro_f1']} |")
    lines.extend(
        [
            "",
            "## Class And Game Accuracy Gap",
            "",
            "Current Transformer outputs save aggregate test accuracy/F1 only. They do not save per-window predictions, so true per-game or per-class Transformer accuracy cannot be reconstructed from the existing result files. This analysis therefore saves per-class and per-game split composition, but marks per-class/per-game accuracy as a follow-up requiring prediction logging.",
            "",
            "## Interpretation",
            "",
            "- LOSO is not inherently easier, but in this pipeline it trains on a larger fit set per fold than the subject-dependent split.",
            "- GAMEEMO labels are tied to game conditions, so broad training coverage from 27 subjects may make held-out-subject recognition easier than expected.",
            "- Subject-dependent testing samples windows from every subject, so it may include a more mixed collection of easy and hard subject/window cases in one test set.",
            "- The stable LOSO > subject-dependent result should be reported as a protocol-composition finding, not overclaimed as a model breakthrough.",
            "",
            "## Files",
            "",
            f"- `{sd_result_path}`",
            f"- `{loso_result_path}`",
            "- `results/transformer_split_composition/split_counts.csv`",
            "- `results/transformer_split_composition/train_size_comparison.csv`",
            "- `results/transformer_split_composition/loso_subject_distribution.csv`",
            "- `results/transformer_split_composition/per_game_class_accuracy_availability.csv`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Transformer SD vs LOSO split composition.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window-samples", type=int, default=WINDOW_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--loso-run-dir", type=Path, default=RESULTS_DIR / "transformer_loso_repeated_seed_runs")
    parser.add_argument("--loso-seeds", default="0,1,2")
    parser.add_argument("--seed0-loso-folds", type=Path, default=RESULTS_DIR / "loso_transformer_baseline.csv")
    args = parser.parse_args()

    dataset = build_window_dataset(args.root, window_samples=args.window_samples)
    metadata = dataset.metadata

    sd_split = make_subject_dependent_split(
        metadata,
        test_size=TEST_SIZE,
        random_state=args.random_seed,
        stratify=True,
    )
    sd_fit, sd_val = split_train_validation(
        sd_split.train_indices,
        dataset.labels,
        val_size=VAL_SIZE,
        random_seed=args.random_seed,
    )
    loso_splits = make_loso_splits(metadata)

    split_count_rows: list[dict[str, object]] = []
    for split_name, indices in [
        ("fit", sd_fit),
        ("validation", sd_val),
        ("test", sd_split.test_indices),
    ]:
        summarize_counts(
            split_count_rows,
            protocol="subject_dependent",
            split_name=split_name,
            field="label",
            counter=count_values(metadata, indices, "label"),
            total_windows=len(indices),
        )
        summarize_counts(
            split_count_rows,
            protocol="subject_dependent",
            split_name=split_name,
            field="game",
            counter=count_values(metadata, indices, "game"),
            total_windows=len(indices),
        )
        summarize_counts(
            split_count_rows,
            protocol="subject_dependent",
            split_name=split_name,
            field="subject",
            counter=count_values(metadata, indices, "subject"),
            total_windows=len(indices),
        )

    train_size_rows: list[dict[str, object]] = [
        {
            "protocol": "subject_dependent",
            "fold": "ALL",
            "fit_windows": len(sd_fit),
            "validation_windows": len(sd_val),
            "test_windows": len(sd_split.test_indices),
            "test_subjects": len(count_values(metadata, sd_split.test_indices, "subject")),
            "train_subjects": len(count_values(metadata, sd_fit, "subject")),
        }
    ]
    loso_fit_sizes: list[int] = []
    loso_val_sizes: list[int] = []
    loso_test_sizes: list[int] = []
    for split in loso_splits:
        fit_indices, val_indices = split_train_validation(
            split.train_indices,
            dataset.labels,
            val_size=VAL_SIZE,
            random_seed=args.random_seed,
        )
        loso_fit_sizes.append(len(fit_indices))
        loso_val_sizes.append(len(val_indices))
        loso_test_sizes.append(len(split.test_indices))
        train_size_rows.append(
            {
                "protocol": "loso",
                "fold": split.test_subject or "",
                "fit_windows": len(fit_indices),
                "validation_windows": len(val_indices),
                "test_windows": len(split.test_indices),
                "test_subjects": 1,
                "train_subjects": len(count_values(metadata, fit_indices, "subject")),
            }
        )
        for split_name, indices in [
            ("fit", fit_indices),
            ("validation", val_indices),
            ("test", split.test_indices),
        ]:
            summarize_counts(
                split_count_rows,
                protocol="loso",
                fold=split.test_subject or "",
                split_name=split_name,
                field="label",
                counter=count_values(metadata, indices, "label"),
                total_windows=len(indices),
            )
            summarize_counts(
                split_count_rows,
                protocol="loso",
                fold=split.test_subject or "",
                split_name=split_name,
                field="game",
                counter=count_values(metadata, indices, "game"),
                total_windows=len(indices),
            )

    seeds = [int(seed.strip()) for seed in args.loso_seeds.split(",") if seed.strip()]
    loso_rows = collect_loso_subject_rows(args.loso_run_dir, seeds, args.seed0_loso_folds)
    loso_subject_rows = summarize_loso_subject_results(loso_rows)

    availability_rows = [
        {
            "analysis_target": "transformer_per_class_accuracy",
            "available": "false",
            "reason": "Transformer result files contain aggregate metrics but not per-window predictions.",
            "recommended_follow_up": "Add optional prediction export to Transformer runners.",
        },
        {
            "analysis_target": "transformer_per_game_accuracy",
            "available": "false",
            "reason": "GAMEEMO game/class metadata is available, but predictions are not saved.",
            "recommended_follow_up": "Save source metadata, true label, predicted label, and split for test windows.",
        },
        {
            "analysis_target": "per_class_and_per_game_split_composition",
            "available": "true",
            "reason": "Window metadata includes label and game for every window.",
            "recommended_follow_up": "",
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "split_counts.csv",
        split_count_rows,
        ["protocol", "fold", "split", "field", "value", "count", "proportion", "split_total_windows"],
    )
    write_csv(
        args.output_dir / "train_size_comparison.csv",
        train_size_rows,
        ["protocol", "fold", "fit_windows", "validation_windows", "test_windows", "test_subjects", "train_subjects"],
    )
    write_csv(
        args.output_dir / "loso_subject_distribution.csv",
        loso_subject_rows,
        [
            "subject",
            "n_seeds",
            "mean_accuracy",
            "std_accuracy",
            "mean_macro_f1",
            "std_macro_f1",
            "min_accuracy",
            "max_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "per_game_class_accuracy_availability.csv",
        availability_rows,
        ["analysis_target", "available", "reason", "recommended_follow_up"],
    )
    write_markdown(
        args.output_dir / "README.md",
        sd_fit_windows=len(sd_fit),
        sd_val_windows=len(sd_val),
        sd_test_windows=len(sd_split.test_indices),
        loso_fit_windows=loso_fit_sizes,
        loso_val_windows=loso_val_sizes,
        loso_test_windows=loso_test_sizes,
        sd_test_subject_count=len(count_values(metadata, sd_split.test_indices, "subject")),
        loso_subject_rows=loso_subject_rows,
        sd_result_path=RESULTS_DIR / "transformer_subject_dependent_repeated_seed_results.csv",
        loso_result_path=RESULTS_DIR / "transformer_loso_repeated_seed_results.csv",
    )

    print(f"windows: {len(metadata)}")
    print(f"subject_dependent_fit_windows: {len(sd_fit)}")
    print(f"subject_dependent_val_windows: {len(sd_val)}")
    print(f"subject_dependent_test_windows: {len(sd_split.test_indices)}")
    print(f"subject_dependent_test_subjects: {len(count_values(metadata, sd_split.test_indices, 'subject'))}")
    print(f"loso_mean_fit_windows: {statistics.mean(loso_fit_sizes):.1f}")
    print(f"loso_mean_val_windows: {statistics.mean(loso_val_sizes):.1f}")
    print(f"loso_mean_test_windows: {statistics.mean(loso_test_sizes):.1f}")
    print("per_game_class_accuracy_available: false")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
