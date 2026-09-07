"""Run repeated-seed temporal-patch Transformer experiments on GAMEEMO."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT


DEFAULT_SEEDS = [0, 1, 2, 3, 4]
OUTPUT_PATH = RESULTS_DIR / "transformer_repeated_seed_results.csv"
RUN_DIR = RESULTS_DIR / "transformer_repeated_seed_runs"


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def write_results(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "protocol",
        "split_protocol",
        "accuracy",
        "macro_f1",
        "accuracy_std",
        "macro_f1_std",
        "n_folds",
        "window_samples",
        "input_mode",
        "patch_samples",
        "learning_rate",
        "dropout",
        "d_model",
        "n_heads",
        "n_layers",
        "dim_feedforward",
        "class_weight",
        "normalization_strategy",
        "requested_epochs",
        "batch_size",
        "weight_decay",
        "patience",
        "device",
        "source_file",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_subject_dependent(args, seed: int) -> dict[str, str]:
    output_path = args.run_dir / f"subject_dependent_seed_{seed}.csv"
    command = [
        args.python_executable,
        "-u",
        "-B",
        str(PROJECT_ROOT / "scripts" / "run_subject_dependent_transformer_baseline.py"),
        "--root",
        str(args.root),
        "--output",
        str(output_path),
        "--random-seed",
        str(seed),
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
        "--input-mode",
        args.input_mode,
        "--patch-samples",
        str(args.patch_samples),
        "--class-weight",
        args.class_weight,
        "--window-samples",
        str(args.window_samples),
        "--device",
        args.device,
    ]
    if args.split_protocol != "subject_dependent":
        command.extend(["--split-protocol", args.split_protocol])
    if args.limit_records is not None:
        command.extend(["--limit-records", str(args.limit_records)])

    print(f"running protocol=subject_dependent split_protocol={args.split_protocol} seed={seed}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    result = read_single_row(output_path)
    return {
        "seed": str(seed),
        "protocol": "subject_dependent",
        "split_protocol": result["split_protocol"],
        "accuracy": result["test_accuracy"],
        "macro_f1": result["test_macro_f1"],
        "accuracy_std": "",
        "macro_f1_std": "",
        "n_folds": "1",
        "window_samples": result["window_samples"],
        "input_mode": result["input_mode"],
        "patch_samples": result["patch_samples"],
        "learning_rate": result["learning_rate"],
        "dropout": result["dropout"],
        "d_model": result["d_model"],
        "n_heads": result["n_heads"],
        "n_layers": result["n_layers"],
        "dim_feedforward": result["dim_feedforward"],
        "class_weight": result["class_weight"],
        "normalization_strategy": result["normalization_strategy"],
        "requested_epochs": result["requested_epochs"],
        "batch_size": result["batch_size"],
        "weight_decay": result["weight_decay"],
        "patience": str(args.patience),
        "device": result["device"],
        "source_file": str(output_path),
    }


def run_loso(args, seed: int) -> dict[str, str]:
    output_path = args.run_dir / f"loso_seed_{seed}.csv"
    summary_path = args.run_dir / f"loso_seed_{seed}_summary.csv"
    command = [
        args.python_executable,
        "-u",
        "-B",
        str(PROJECT_ROOT / "scripts" / "run_loso_transformer_baseline.py"),
        "--root",
        str(args.root),
        "--output",
        str(output_path),
        "--summary-output",
        str(summary_path),
        "--random-seed",
        str(seed),
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
        "--input-mode",
        args.input_mode,
        "--patch-samples",
        str(args.patch_samples),
        "--class-weight",
        args.class_weight,
        "--window-samples",
        str(args.window_samples),
        "--device",
        args.device,
        "--resume-existing",
    ]
    if args.max_folds is not None:
        command.extend(["--max-folds", str(args.max_folds)])
    if args.limit_records is not None:
        command.extend(["--limit-records", str(args.limit_records)])

    print(f"running protocol=loso seed={seed}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    result = read_single_row(summary_path)
    return {
        "seed": str(seed),
        "protocol": "loso",
        "split_protocol": "loso",
        "accuracy": result["mean_accuracy"],
        "macro_f1": result["mean_macro_f1"],
        "accuracy_std": result["std_accuracy"],
        "macro_f1_std": result["std_macro_f1"],
        "n_folds": result["n_folds"],
        "window_samples": args.window_samples,
        "input_mode": result["input_mode"],
        "patch_samples": result["patch_samples"],
        "learning_rate": result["learning_rate"],
        "dropout": result["dropout"],
        "d_model": result["d_model"],
        "n_heads": result["n_heads"],
        "n_layers": result["n_layers"],
        "dim_feedforward": result["dim_feedforward"],
        "class_weight": result["class_weight"],
        "normalization_strategy": result["normalization_strategy"],
        "requested_epochs": result["requested_epochs"],
        "batch_size": result["batch_size"],
        "weight_decay": result["weight_decay"],
        "patience": str(args.patience),
        "device": result["device"],
        "source_file": str(summary_path),
    }


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in raw.split(",") if seed.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def write_summary(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "protocol",
        "split_protocol",
        "n_seeds",
        "seeds",
        "mean_accuracy",
        "std_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
        "window_samples",
        "input_mode",
        "patch_samples",
        "learning_rate",
        "dropout",
        "d_model",
        "n_heads",
        "n_layers",
        "dim_feedforward",
        "class_weight",
        "normalization_strategy",
        "requested_epochs",
        "batch_size",
        "weight_decay",
        "patience",
    ]
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["protocol"], row["split_protocol"]), []).append(row)

    summary_rows = []
    for (protocol, split_protocol), group_rows in sorted(groups.items()):
        accuracies = [float(row["accuracy"]) for row in group_rows]
        macro_f1s = [float(row["macro_f1"]) for row in group_rows]
        first = group_rows[0]
        summary_rows.append(
            {
                "model": "TemporalPatchTransformer" if first["input_mode"] == "temporal_patch" else "ChannelTokenTransformer",
                "protocol": protocol,
                "split_protocol": split_protocol,
                "n_seeds": len(group_rows),
                "seeds": ",".join(row["seed"] for row in group_rows),
                "mean_accuracy": f"{statistics.mean(accuracies):.6f}",
                "std_accuracy": f"{statistics.stdev(accuracies):.6f}" if len(accuracies) > 1 else "0.000000",
                "mean_macro_f1": f"{statistics.mean(macro_f1s):.6f}",
                "std_macro_f1": f"{statistics.stdev(macro_f1s):.6f}" if len(macro_f1s) > 1 else "0.000000",
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
        )

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-seed GAMEEMO Transformer experiments.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable to use for launched training jobs. Defaults to the current interpreter.",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--protocol", choices=["subject_dependent", "loso", "both"], default="both")
    parser.add_argument(
        "--split-protocol",
        choices=["subject_dependent", "subject_dependent_matched_budget"],
        default="subject_dependent",
        help="Subject-dependent split protocol to use when protocol includes subject_dependent.",
    )
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
    parser.add_argument("--input-mode", choices=["channel", "temporal_patch"], default="temporal_patch")
    parser.add_argument("--patch-samples", type=int, default=32)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--window-samples", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional LOSO fold limit for smoke tests.")
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise SystemExit(f"batch-size must be positive, got {args.batch_size}")
    if args.patience <= 0:
        raise SystemExit(f"patience must be positive, got {args.patience}")
    if args.window_samples <= 0:
        raise SystemExit(f"window-samples must be positive, got {args.window_samples}")
    if args.patch_samples <= 0:
        raise SystemExit(f"patch-samples must be positive, got {args.patch_samples}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for seed in args.seeds:
        if args.protocol in {"subject_dependent", "both"}:
            rows.append(run_subject_dependent(args, seed))
            write_results(args.output, rows)
            print(f"wrote_partial: {args.output}")
        if args.protocol in {"loso", "both"}:
            rows.append(run_loso(args, seed))
            write_results(args.output, rows)
            print(f"wrote_partial: {args.output}")

    write_results(args.output, rows)
    print(f"wrote: {args.output}")
    if args.summary_output is not None:
        write_summary(args.summary_output, rows)
        print(f"wrote_summary: {args.summary_output}")


if __name__ == "__main__":
    main()
