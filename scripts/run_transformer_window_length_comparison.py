"""Compare 2-second and 4-second Transformer windows on GAMEEMO."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, SAMPLING_RATE
from src.gameemo_loader import GAMEEMO_ROOT


WINDOW_SECONDS = [2, 4]
OUTPUT_PATH = RESULTS_DIR / "transformer_window_length_comparison.csv"
RUN_DIR = RESULTS_DIR / "transformer_window_length_runs"


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def write_comparison(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_seconds",
        "window_samples",
        "test_accuracy",
        "test_macro_f1",
        "best_val_accuracy",
        "best_val_macro_f1",
        "best_epoch",
        "epochs_ran",
        "requested_epochs",
        "n_windows",
        "train_windows",
        "val_windows",
        "test_windows",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
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


def run_window_length(args, window_seconds: int) -> dict[str, str]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    window_samples = window_seconds * args.sampling_rate
    combo_output = args.run_dir / f"window_{window_seconds}s.csv"

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
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--dropout",
        str(args.dropout),
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
        "--window-samples",
        str(window_samples),
    ]
    if args.limit_records is not None:
        command.extend(["--limit-records", str(args.limit_records)])

    print(f"running window_seconds={window_seconds} window_samples={window_samples}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    result = read_single_row(combo_output)
    return {
        "window_seconds": str(window_seconds),
        "window_samples": result["window_samples"],
        "test_accuracy": result["test_accuracy"],
        "test_macro_f1": result["test_macro_f1"],
        "best_val_accuracy": result["best_val_accuracy"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "requested_epochs": result["requested_epochs"],
        "n_windows": result["n_windows"],
        "train_windows": result["train_windows"],
        "val_windows": result["val_windows"],
        "test_windows": result["test_windows"],
        "batch_size": result["batch_size"],
        "learning_rate": result["learning_rate"],
        "weight_decay": result["weight_decay"],
        "dropout": result["dropout"],
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
    parser = argparse.ArgumentParser(description="Compare GAMEEMO Transformer 2s and 4s windows.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--sampling-rate", type=int, default=SAMPLING_RATE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    if args.sampling_rate <= 0:
        raise SystemExit(f"sampling-rate must be positive, got {args.sampling_rate}")

    rows = []
    for window_seconds in WINDOW_SECONDS:
        rows.append(run_window_length(args, window_seconds))
        write_comparison(args.output, rows)
        print(f"wrote_partial: {args.output}")

    write_comparison(args.output, rows)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
