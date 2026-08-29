"""Run a small EEGNet learning-rate/dropout grid on GAMEEMO."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT


LEARNING_RATES = [0.001, 0.0005]
DROPOUTS = [0.25, 0.5]
OUTPUT_PATH = RESULTS_DIR / "eegnet_lr_dropout_grid.csv"
RUN_DIR = RESULTS_DIR / "eegnet_lr_dropout_grid_runs"


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def write_grid(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "learning_rate",
        "dropout",
        "test_accuracy",
        "test_macro_f1",
        "best_val_accuracy",
        "best_val_macro_f1",
        "best_epoch",
        "epochs_ran",
        "requested_epochs",
        "batch_size",
        "weight_decay",
        "patience",
        "temporal_filters",
        "depth_multiplier",
        "separable_filters",
        "temporal_kernel",
        "separable_kernel",
        "device",
        "source_file",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_combo(args, learning_rate: float, dropout: float) -> dict[str, str]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    combo_name = f"lr_{learning_rate:g}_dropout_{dropout:g}".replace(".", "p")
    combo_output = args.run_dir / f"{combo_name}.csv"

    command = [
        sys.executable,
        "-u",
        "-B",
        str(PROJECT_ROOT / "scripts" / "run_subject_dependent_eegnet_baseline.py"),
        "--root",
        str(args.root),
        "--output",
        str(combo_output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--dropout",
        str(dropout),
        "--temporal-filters",
        str(args.temporal_filters),
        "--depth-multiplier",
        str(args.depth_multiplier),
        "--separable-filters",
        str(args.separable_filters),
        "--temporal-kernel",
        str(args.temporal_kernel),
        "--separable-kernel",
        str(args.separable_kernel),
        "--device",
        args.device,
    ]
    if args.limit_records is not None:
        command.extend(["--limit-records", str(args.limit_records)])

    print(f"running learning_rate={learning_rate} dropout={dropout}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    result = read_single_row(combo_output)
    return {
        "learning_rate": result["learning_rate"],
        "dropout": result["dropout"],
        "test_accuracy": result["test_accuracy"],
        "test_macro_f1": result["test_macro_f1"],
        "best_val_accuracy": result["best_val_accuracy"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "requested_epochs": result["requested_epochs"],
        "batch_size": result["batch_size"],
        "weight_decay": result["weight_decay"],
        "patience": str(args.patience),
        "temporal_filters": result["temporal_filters"],
        "depth_multiplier": result["depth_multiplier"],
        "separable_filters": result["separable_filters"],
        "temporal_kernel": result["temporal_kernel"],
        "separable_kernel": result["separable_kernel"],
        "device": result["device"],
        "source_file": str(combo_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2x2 EEGNet learning-rate/dropout grid.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--temporal-filters", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--separable-filters", type=int, default=16)
    parser.add_argument("--temporal-kernel", type=int, default=64)
    parser.add_argument("--separable-kernel", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for learning_rate in LEARNING_RATES:
        for dropout in DROPOUTS:
            rows.append(run_combo(args, learning_rate, dropout))
            write_grid(args.output, rows)
            print(f"wrote_partial: {args.output}")

    write_grid(args.output, rows)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
