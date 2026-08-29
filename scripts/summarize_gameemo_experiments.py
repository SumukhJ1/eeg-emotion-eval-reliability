"""Create a consolidated GAMEEMO experiment summary table."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR


CHANCE_ACCURACY = 0.25
OUTPUT_PATH = RESULTS_DIR / "gameemo_experiment_summary.csv"


@dataclass(frozen=True)
class ExperimentRow:
    feature_input: str
    model_name: str
    protocol: str
    accuracy: float
    macro_f1: float
    chance_accuracy: float = CHANCE_ACCURACY
    protocol_drop_accuracy: float | None = None
    protocol_drop_macro_f1: float | None = None
    source_file: str = ""


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def find_row(rows: list[dict[str, str]], *, model_name: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("model_name") == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for model_name={model_name}, got {len(matches)}")
    return matches[0]


def add_protocol_drops(rows: list[ExperimentRow]) -> list[ExperimentRow]:
    by_key = {(row.feature_input, row.model_name, row.protocol): row for row in rows}
    updated: list[ExperimentRow] = []

    for row in rows:
        if row.protocol != "subject_dependent":
            updated.append(row)
            continue

        loso = by_key.get((row.feature_input, row.model_name, "loso"))
        if loso is None:
            updated.append(row)
            continue

        updated.append(
            ExperimentRow(
                feature_input=row.feature_input,
                model_name=row.model_name,
                protocol=row.protocol,
                accuracy=row.accuracy,
                macro_f1=row.macro_f1,
                chance_accuracy=row.chance_accuracy,
                protocol_drop_accuracy=row.accuracy - loso.accuracy,
                protocol_drop_macro_f1=row.macro_f1 - loso.macro_f1,
                source_file=row.source_file,
            )
        )

    return updated


def build_summary_rows(results_dir: Path = RESULTS_DIR) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []

    stat_lr = read_single_row(results_dir / "subject_dependent_baseline.csv")
    rows.append(
        ExperimentRow(
            feature_input="time_statistical",
            model_name="LogisticRegression",
            protocol="subject_dependent",
            accuracy=float(stat_lr["accuracy"]),
            macro_f1=float(stat_lr["macro_f1"]),
            source_file="results/subject_dependent_baseline.csv",
        )
    )

    stat_svm = read_single_row(results_dir / "subject_dependent_svm_baseline.csv")
    rows.append(
        ExperimentRow(
            feature_input="time_statistical",
            model_name="LinearSVM",
            protocol="subject_dependent",
            accuracy=float(stat_svm["accuracy"]),
            macro_f1=float(stat_svm["macro_f1"]),
            source_file="results/subject_dependent_svm_baseline.csv",
        )
    )

    stat_loso_lr = read_single_row(results_dir / "loso_summary.csv")
    rows.append(
        ExperimentRow(
            feature_input="time_statistical",
            model_name="LogisticRegression",
            protocol="loso",
            accuracy=float(stat_loso_lr["mean_accuracy"]),
            macro_f1=float(stat_loso_lr["mean_macro_f1"]),
            source_file="results/loso_summary.csv",
        )
    )

    stat_loso_svm = read_single_row(results_dir / "loso_svm_summary.csv")
    rows.append(
        ExperimentRow(
            feature_input="time_statistical",
            model_name="LinearSVM",
            protocol="loso",
            accuracy=float(stat_loso_svm["mean_accuracy"]),
            macro_f1=float(stat_loso_svm["mean_macro_f1"]),
            source_file="results/loso_svm_summary.csv",
        )
    )

    log_relative_subject_rows = read_rows(results_dir / "subject_dependent_log_relative_bandpower_baselines.csv")
    for model_name in ["LogisticRegression", "LinearSVM"]:
        row = find_row(log_relative_subject_rows, model_name=model_name)
        rows.append(
            ExperimentRow(
                feature_input="log_relative_bandpower",
                model_name=model_name,
                protocol="subject_dependent",
                accuracy=float(row["accuracy"]),
                macro_f1=float(row["macro_f1"]),
                source_file="results/subject_dependent_log_relative_bandpower_baselines.csv",
            )
        )

    log_relative_loso_rows = read_rows(results_dir / "loso_log_relative_bandpower_summary.csv")
    for model_name in ["LogisticRegression", "LinearSVM"]:
        row = find_row(log_relative_loso_rows, model_name=model_name)
        rows.append(
            ExperimentRow(
                feature_input="log_relative_bandpower",
                model_name=model_name,
                protocol="loso",
                accuracy=float(row["mean_accuracy"]),
                macro_f1=float(row["mean_macro_f1"]),
                source_file="results/loso_log_relative_bandpower_summary.csv",
            )
        )

    eegnet = read_single_row(results_dir / "subject_dependent_eegnet_baseline.csv")
    rows.append(
        ExperimentRow(
            feature_input="raw_eeg_windows",
            model_name="EEGNet",
            protocol="subject_dependent",
            accuracy=float(eegnet["test_accuracy"]),
            macro_f1=float(eegnet["test_macro_f1"]),
            source_file="results/subject_dependent_eegnet_baseline.csv",
        )
    )

    return add_protocol_drops(rows)


def format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_summary(output_path: Path, rows: list[ExperimentRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_input",
        "model_name",
        "protocol",
        "accuracy",
        "macro_f1",
        "chance_accuracy",
        "accuracy_above_chance",
        "macro_f1_above_chance",
        "protocol_drop_accuracy",
        "protocol_drop_macro_f1",
        "source_file",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "feature_input": row.feature_input,
                    "model_name": row.model_name,
                    "protocol": row.protocol,
                    "accuracy": f"{row.accuracy:.6f}",
                    "macro_f1": f"{row.macro_f1:.6f}",
                    "chance_accuracy": f"{row.chance_accuracy:.6f}",
                    "accuracy_above_chance": f"{row.accuracy - row.chance_accuracy:.6f}",
                    "macro_f1_above_chance": f"{row.macro_f1 - row.chance_accuracy:.6f}",
                    "protocol_drop_accuracy": format_optional(row.protocol_drop_accuracy),
                    "protocol_drop_macro_f1": format_optional(row.protocol_drop_macro_f1),
                    "source_file": row.source_file,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a consolidated GAMEEMO experiment summary CSV.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    rows = build_summary_rows(args.results_dir)
    write_summary(args.output, rows)

    print(f"wrote: {args.output}")
    for row in rows:
        drop = format_optional(row.protocol_drop_accuracy)
        print(
            f"{row.protocol} {row.feature_input} {row.model_name}: "
            f"accuracy={row.accuracy:.6f} macro_f1={row.macro_f1:.6f} "
            f"chance={row.chance_accuracy:.6f} protocol_drop_accuracy={drop}"
        )


if __name__ == "__main__":
    main()
