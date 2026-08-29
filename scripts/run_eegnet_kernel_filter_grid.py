"""Run a small EEGNet temporal-kernel/F1-filter grid on GAMEEMO."""

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


TEMPORAL_KERNELS = [32, 64]
TEMPORAL_FILTERS = [4, 8]
OUTPUT_PATH = RESULTS_DIR / "eegnet_kernel_filter_grid.csv"
RUN_DIR = RESULTS_DIR / "eegnet_kernel_filter_grid_runs"


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def write_grid(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "temporal_kernel",
        "temporal_filters",
        "test_accuracy",
        "test_macro_f1",
        "best_val_accuracy",
        "best_val_macro_f1",
        "best_epoch",
        "epochs_ran",
        "requested_epochs",
        "learning_rate",
        "dropout",
        "batch_size",
        "weight_decay",
        "patience",
        "depth_multiplier",
        "separable_filters",
        "separable_kernel",
        "device",
        "source_file",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_combo(args, temporal_kernel: int, temporal_filters: int) -> dict[str, str]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    combo_name = f"kernel_{temporal_kernel}_filters_{temporal_filters}"
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
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--dropout",
        str(args.dropout),
        "--temporal-filters",
        str(temporal_filters),
        "--depth-multiplier",
        str(args.depth_multiplier),
        "--separable-filters",
        str(args.separable_filters),
        "--temporal-kernel",
        str(temporal_kernel),
        "--separable-kernel",
        str(args.separable_kernel),
        "--device",
        args.device,
    ]
    if args.limit_records is not None:
        command.extend(["--limit-records", str(args.limit_records)])

    print(f"running temporal_kernel={temporal_kernel} temporal_filters={temporal_filters}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    result = read_single_row(combo_output)
    return {
        "temporal_kernel": result["temporal_kernel"],
        "temporal_filters": result["temporal_filters"],
        "test_accuracy": result["test_accuracy"],
        "test_macro_f1": result["test_macro_f1"],
        "best_val_accuracy": result["best_val_accuracy"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "requested_epochs": result["requested_epochs"],
        "learning_rate": result["learning_rate"],
        "dropout": result["dropout"],
        "batch_size": result["batch_size"],
        "weight_decay": result["weight_decay"],
        "patience": str(args.patience),
        "depth_multiplier": result["depth_multiplier"],
        "separable_filters": result["separable_filters"],
        "separable_kernel": result["separable_kernel"],
        "device": result["device"],
        "source_file": str(combo_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2x2 EEGNet temporal-kernel/F1-filter grid.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--separable-filters", type=int, default=16)
    parser.add_argument("--separable-kernel", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for temporal_kernel in TEMPORAL_KERNELS:
        for temporal_filters in TEMPORAL_FILTERS:
            rows.append(run_combo(args, temporal_kernel, temporal_filters))
            write_grid(args.output, rows)
            print(f"wrote_partial: {args.output}")

    write_grid(args.output, rows)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
