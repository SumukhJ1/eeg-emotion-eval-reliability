"""Summarize repeated-seed Transformer LOSO verification results."""

from __future__ import annotations

import argparse
import csv
import sys
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR


DEFAULT_INPUT = RESULTS_DIR / "transformer_loso_repeated_seed_results.csv"
DEFAULT_SUMMARY = RESULTS_DIR / "transformer_loso_repeated_seed_summary.csv"


def read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    rows = [row for row in rows if row.get("protocol") == "loso"]
    if not rows:
        raise SystemExit(f"No LOSO rows found in {input_path}")
    return rows


def write_summary(output_path: Path, rows: list[dict[str, str]]) -> None:
    accuracies = [float(row["accuracy"]) for row in rows]
    macro_f1s = [float(row["macro_f1"]) for row in rows]
    first = rows[0]
    summary = {
        "model": "TemporalPatchTransformer",
        "protocol": "loso",
        "n_seeds": len(rows),
        "seeds": ",".join(row["seed"] for row in rows),
        "mean_accuracy": f"{statistics.mean(accuracies):.6f}",
        "std_accuracy": f"{statistics.pstdev(accuracies):.6f}",
        "mean_macro_f1": f"{statistics.mean(macro_f1s):.6f}",
        "std_macro_f1": f"{statistics.pstdev(macro_f1s):.6f}",
        "window_samples": first["window_samples"],
        "input_mode": first["input_mode"],
        "patch_samples": first["patch_samples"],
        "learning_rate": first["learning_rate"],
        "dropout": first["dropout"],
        "d_model": first["d_model"],
        "n_heads": first["n_heads"],
        "n_layers": first["n_layers"],
        "dim_feedforward": first["dim_feedforward"],
        "class_weight": first["class_weight"],
        "normalization_strategy": first["normalization_strategy"],
        "requested_epochs": first["requested_epochs"],
        "batch_size": first["batch_size"],
        "weight_decay": first["weight_decay"],
        "patience": first["patience"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def print_markdown_tables(rows: list[dict[str, str]], summary_path: Path) -> None:
    print("| Seed | LOSO accuracy | LOSO macro-F1 |")
    print("| --- | --- | --- |")
    for row in rows:
        print(f"| {row['seed']} | {float(row['accuracy']):.6f} | {float(row['macro_f1']):.6f} |")
    with summary_path.open(newline="") as csv_file:
        summary = next(csv.DictReader(csv_file))
    print()
    print("| Model | Mean acc | Std acc | Mean F1 | Std F1 |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| {summary['model']} | {float(summary['mean_accuracy']):.6f} | "
        f"{float(summary['std_accuracy']):.6f} | {float(summary['mean_macro_f1']):.6f} | "
        f"{float(summary['std_macro_f1']):.6f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated-seed Transformer LOSO results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_rows(args.input)
    write_summary(args.summary_output, rows)
    print_markdown_tables(rows, args.summary_output)
    print(f"wrote: {args.summary_output}")


if __name__ == "__main__":
    main()
