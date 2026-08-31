"""Run a small Transformer learning-rate/dropout grid on GAMEEMO."""

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
DROPOUTS = [0.1, 0.25]
OUTPUT_PATH = RESULTS_DIR / "transformer_lr_dropout_grid.csv"
RUN_DIR = RESULTS_DIR / "transformer_lr_dropout_grid_runs"


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
        "d_model",
        "n_heads",
        "n_layers",
        "dim_feedforward",
        "device",
        "normalization_strategy",
        "status",
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
        str(PROJECT_ROOT / "scripts" / "run_subject_dependent_transformer_baseline.py"),
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
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--n-layers",
        str(args.n_layers),
        "--dim-feedforward",
        str(args.dim_feedforward),
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
        "d_model": result["d_model"],
        "n_heads": result["n_heads"],
        "n_layers": result["n_layers"],
        "dim_feedforward": result["dim_feedforward"],
        "device": result["device"],
        "normalization_strategy": result["normalization_strategy"],
        "status": result["status"],
        "source_file": str(combo_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2x2 Transformer learning-rate/dropout grid.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
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
