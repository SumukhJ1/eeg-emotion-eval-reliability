"""Run leave-one-subject-out EEGNet baselines on GAMEEMO windows."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_window_dataset
from src.config import LABEL_MAP, RESULTS_DIR
from src.eegnet import EEGNetConfig, build_eegnet
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import MetadataSplit, make_loso_splits, validate_loso_split
from src.windowing import DEFAULT_WINDOW_SAMPLES, EegWindowMetadata


RANDOM_SEED = 0
VAL_SIZE = 0.2
RESULTS_PATH = RESULTS_DIR / "loso_eegnet_baseline.csv"
SUMMARY_PATH = RESULTS_DIR / "loso_eegnet_summary.csv"


@dataclass(frozen=True)
class FoldResult:
    subject: str
    train_windows: int
    val_windows: int
    test_windows: int
    requested_epochs: int
    epochs_ran: int
    best_epoch: int
    best_val_accuracy: float
    best_val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    final_train_loss: float
    best_train_loss: float
    early_stopped: bool
    dropout: float
    temporal_filters: int
    depth_multiplier: int
    separable_filters: int
    temporal_kernel: int
    separable_kernel: int


class TorchDevice:
    def __init__(self, torch_module, requested: str) -> None:
        if requested == "auto":
            requested = "cuda" if torch_module.cuda.is_available() else "cpu"
        self.value = torch_module.device(requested)
        self._torch = torch_module

    def no_grad_context(self):
        return self._torch.no_grad()


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the EEGNet LOSO baseline. Install torch before running this script."
        ) from exc
    return torch, DataLoader, TensorDataset


def assert_no_subject_leakage(
    metadata: list[EegWindowMetadata],
    split: MetadataSplit,
    train_indices: list[int],
    val_indices: list[int],
) -> None:
    validate_loso_split(metadata, split)
    heldout = split.test_subject
    train_subjects = {metadata[idx].subject for idx in train_indices}
    val_subjects = {metadata[idx].subject for idx in val_indices}
    test_subjects = {metadata[idx].subject for idx in split.test_indices}

    if test_subjects != {heldout}:
        raise ValueError(f"Expected only held-out subject {heldout}, got {sorted(test_subjects)}")
    if heldout in train_subjects:
        raise ValueError(f"LOSO leakage: held-out subject {heldout} appears in training windows")
    if heldout in val_subjects:
        raise ValueError(f"LOSO leakage: held-out subject {heldout} appears in validation windows")


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


def channel_standardize_splits(windows, train_indices, val_indices, test_indices):
    train_block = windows[train_indices]
    mean = train_block.mean(axis=(0, 2), keepdims=True)
    std = train_block.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return (
        (windows[train_indices] - mean) / std,
        (windows[val_indices] - mean) / std,
        (windows[test_indices] - mean) / std,
    )


def make_loader(torch, DataLoader, TensorDataset, windows, labels, batch_size: int, shuffle: bool):
    return DataLoader(
        TensorDataset(
            torch.as_tensor(windows, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * inputs.shape[0]
        total_items += inputs.shape[0]
    return total_loss / max(total_items, 1)


def evaluate(model, loader, device) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    predictions = []
    targets = []
    with device.no_grad_context():
        for inputs, batch_targets in loader:
            logits = model(inputs.to(device.value))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(batch_targets.tolist())
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro")
    return float(accuracy), float(macro_f1)


def run_fold(
    *,
    torch,
    DataLoader,
    TensorDataset,
    windows,
    labels,
    metadata: list[EegWindowMetadata],
    split: MetadataSplit,
    device: TorchDevice,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    val_size: float,
    dropout: float,
    temporal_filters: int,
    depth_multiplier: int,
    separable_filters: int,
    temporal_kernel: int,
    separable_kernel: int,
) -> FoldResult:
    train_indices, val_indices = split_train_validation(
        split.train_indices,
        labels,
        val_size=val_size,
        random_seed=RANDOM_SEED,
    )
    assert_no_subject_leakage(metadata, split, train_indices, val_indices)

    x_train, x_val, x_test = channel_standardize_splits(
        windows,
        train_indices,
        val_indices,
        split.test_indices,
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[split.test_indices]

    model = build_eegnet(
        EEGNetConfig(
            n_channels=x_train.shape[1],
            n_samples=x_train.shape[2],
            n_classes=len(LABEL_MAP),
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            temporal_kernel=temporal_kernel,
            separable_kernel=separable_kernel,
            dropout=dropout,
        )
    ).to(device.value)
    train_loader = make_loader(torch, DataLoader, TensorDataset, x_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(torch, DataLoader, TensorDataset, x_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(torch, DataLoader, TensorDataset, x_test, y_test, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    final_loss = 0.0
    best_train_loss = 0.0
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_macro_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        final_loss = train_one_epoch(model, train_loader, optimizer, criterion, device.value)
        val_accuracy, val_macro_f1 = evaluate(model, val_loader, device)
        print(
            f"subject={split.test_subject} epoch={epoch} train_loss={final_loss:.6f} "
            f"val_accuracy={val_accuracy:.6f} val_macro_f1={val_macro_f1:.6f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_epoch = epoch
            best_val_accuracy = val_accuracy
            best_val_macro_f1 = val_macro_f1
            best_train_loss = final_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_stopped = True
            print(f"subject={split.test_subject} early_stopping_epoch={epoch}")
            break

    epochs_ran = epoch
    model.load_state_dict(best_state)
    test_accuracy, test_macro_f1 = evaluate(model, test_loader, device)

    return FoldResult(
        subject=split.test_subject or "",
        train_windows=len(train_indices),
        val_windows=len(val_indices),
        test_windows=len(split.test_indices),
        requested_epochs=epochs,
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_val_macro_f1=best_val_macro_f1,
        test_accuracy=test_accuracy,
        test_macro_f1=test_macro_f1,
        final_train_loss=final_loss,
        best_train_loss=best_train_loss,
        early_stopped=early_stopped,
        dropout=dropout,
        temporal_filters=temporal_filters,
        depth_multiplier=depth_multiplier,
        separable_filters=separable_filters,
        temporal_kernel=temporal_kernel,
        separable_kernel=separable_kernel,
    )


def result_fieldnames() -> list[str]:
    return [
        "experiment",
        "model_name",
        "subject",
        "random_seed",
        "val_size",
        "window_samples",
        "n_records",
        "n_windows",
        "n_channels",
        "n_classes",
        "train_windows",
        "val_windows",
        "test_windows",
        "requested_epochs",
        "epochs_ran",
        "best_epoch",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "temporal_filters",
        "depth_multiplier",
        "separable_filters",
        "temporal_kernel",
        "separable_kernel",
        "device",
        "best_val_accuracy",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "final_train_loss",
        "best_train_loss",
        "early_stopped",
    ]


def format_result_row(
    result: FoldResult,
    *,
    window_samples: int,
    n_records: int,
    n_windows: int,
    n_channels: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    val_size: float,
) -> dict[str, object]:
    return {
        "experiment": "loso_eegnet_raw_windows",
        "model_name": "EEGNet",
        "subject": result.subject,
        "random_seed": RANDOM_SEED,
        "val_size": val_size,
        "window_samples": window_samples,
        "n_records": n_records,
        "n_windows": n_windows,
        "n_channels": n_channels,
        "n_classes": len(LABEL_MAP),
        "train_windows": result.train_windows,
        "val_windows": result.val_windows,
        "test_windows": result.test_windows,
        "requested_epochs": result.requested_epochs,
        "epochs_ran": result.epochs_ran,
        "best_epoch": result.best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": result.dropout,
        "temporal_filters": result.temporal_filters,
        "depth_multiplier": result.depth_multiplier,
        "separable_filters": result.separable_filters,
        "temporal_kernel": result.temporal_kernel,
        "separable_kernel": result.separable_kernel,
        "device": device,
        "best_val_accuracy": f"{result.best_val_accuracy:.6f}",
        "best_val_macro_f1": f"{result.best_val_macro_f1:.6f}",
        "test_accuracy": f"{result.test_accuracy:.6f}",
        "test_macro_f1": f"{result.test_macro_f1:.6f}",
        "final_train_loss": f"{result.final_train_loss:.6f}",
        "best_train_loss": f"{result.best_train_loss:.6f}",
        "early_stopped": result.early_stopped,
    }


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(output_path: Path) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    with output_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    if reader.fieldnames != result_fieldnames():
        raise ValueError(f"Existing output has unexpected columns: {output_path}")
    return rows


def write_summary(output_path: Path, rows: list[dict[str, object]]) -> None:
    import statistics

    accuracies = [float(row["test_accuracy"]) for row in rows]
    macro_f1s = [float(row["test_macro_f1"]) for row in rows]
    best_accuracy = max(rows, key=lambda row: float(row["test_accuracy"]))
    worst_accuracy = min(rows, key=lambda row: float(row["test_accuracy"]))
    best_macro_f1 = max(rows, key=lambda row: float(row["test_macro_f1"]))
    worst_macro_f1 = min(rows, key=lambda row: float(row["test_macro_f1"]))

    summary = {
        "experiment": "loso_eegnet_raw_windows",
        "model_name": "EEGNet",
        "n_folds": len(rows),
        "mean_accuracy": f"{statistics.mean(accuracies):.6f}",
        "std_accuracy": f"{statistics.pstdev(accuracies):.6f}",
        "mean_macro_f1": f"{statistics.mean(macro_f1s):.6f}",
        "std_macro_f1": f"{statistics.pstdev(macro_f1s):.6f}",
        "best_accuracy_subject": best_accuracy["subject"],
        "best_accuracy": best_accuracy["test_accuracy"],
        "worst_accuracy_subject": worst_accuracy["subject"],
        "worst_accuracy": worst_accuracy["test_accuracy"],
        "best_macro_f1_subject": best_macro_f1["subject"],
        "best_macro_f1": best_macro_f1["test_macro_f1"],
        "worst_macro_f1_subject": worst_macro_f1["subject"],
        "worst_macro_f1": worst_macro_f1["test_macro_f1"],
        "requested_epochs": rows[0]["requested_epochs"],
        "batch_size": rows[0]["batch_size"],
        "learning_rate": rows[0]["learning_rate"],
        "weight_decay": rows[0]["weight_decay"],
        "dropout": rows[0]["dropout"],
        "temporal_filters": rows[0]["temporal_filters"],
        "depth_multiplier": rows[0]["depth_multiplier"],
        "separable_filters": rows[0]["separable_filters"],
        "temporal_kernel": rows[0]["temporal_kernel"],
        "separable_kernel": rows[0]["separable_kernel"],
        "device": rows[0]["device"],
    }
    fieldnames = list(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAMEEMO leave-one-subject-out EEGNet baselines.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Path to write per-subject fold results.")
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH, help="Path to write aggregate LOSO metrics.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--temporal-filters", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--separable-filters", type=int, default=16)
    parser.add_argument("--temporal-kernel", type=int, default=64)
    parser.add_argument("--separable-kernel", type=int, default=16)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional debugging limit on LOSO folds.")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Load existing output rows and skip completed held-out subjects.",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help="Fixed window length in samples. Defaults to 2 seconds at 128 Hz.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debugging limit on the number of recordings to load.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise SystemExit(f"batch-size must be positive, got {args.batch_size}")
    if not 0 < args.val_size < 1:
        raise SystemExit(f"val-size must be between 0 and 1, got {args.val_size}")
    if not 0 <= args.dropout < 1:
        raise SystemExit(f"dropout must be in [0, 1), got {args.dropout}")
    if args.temporal_filters <= 0:
        raise SystemExit(f"temporal-filters must be positive, got {args.temporal_filters}")
    if args.depth_multiplier <= 0:
        raise SystemExit(f"depth-multiplier must be positive, got {args.depth_multiplier}")
    if args.separable_filters <= 0:
        raise SystemExit(f"separable-filters must be positive, got {args.separable_filters}")
    if args.temporal_kernel <= 0:
        raise SystemExit(f"temporal-kernel must be positive, got {args.temporal_kernel}")
    if args.separable_kernel <= 0:
        raise SystemExit(f"separable-kernel must be positive, got {args.separable_kernel}")

    torch, DataLoader, TensorDataset = require_torch()
    import numpy as np

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset = build_window_dataset(
        args.root,
        limit_records=args.limit_records,
        window_samples=args.window_samples,
    )
    splits = make_loso_splits(dataset.metadata)
    if args.max_folds is not None:
        splits = splits[: args.max_folds]
    if not splits:
        raise SystemExit("No LOSO folds available.")

    device = TorchDevice(torch, args.device)
    rows: list[dict[str, object]] = read_existing_rows(args.output) if args.resume_existing else []
    completed_subjects = {str(row["subject"]) for row in rows}
    splits = [split for split in splits if split.test_subject not in completed_subjects]
    print(f"records: {dataset.n_records}")
    print(f"windows: {dataset.windows.shape}")
    print(f"folds: {len(splits)}")
    print(f"device: {device.value}")
    if completed_subjects:
        print(f"resumed_completed_subjects: {','.join(sorted(completed_subjects))}")
    if args.resume_existing and not splits:
        write_summary(args.summary_output, rows)
        print(f"wrote: {args.summary_output}")
        return

    for split in splits:
        result = run_fold(
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            windows=dataset.windows,
            labels=dataset.labels,
            metadata=dataset.metadata,
            split=split,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            val_size=args.val_size,
            dropout=args.dropout,
            temporal_filters=args.temporal_filters,
            depth_multiplier=args.depth_multiplier,
            separable_filters=args.separable_filters,
            temporal_kernel=args.temporal_kernel,
            separable_kernel=args.separable_kernel,
        )
        rows.append(
            format_result_row(
                result,
                window_samples=args.window_samples,
                n_records=dataset.n_records,
                n_windows=len(dataset.metadata),
                n_channels=dataset.windows.shape[1],
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=str(device.value),
                val_size=args.val_size,
            )
        )
        write_rows(args.output, rows)
        write_summary(args.summary_output, rows)
        print(
            f"subject={result.subject} test_accuracy={result.test_accuracy:.6f} "
            f"test_macro_f1={result.test_macro_f1:.6f} best_epoch={result.best_epoch}"
        )
        print(f"wrote_partial: {args.output}")
        print(f"wrote_summary: {args.summary_output}")

    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")


if __name__ == "__main__":
    main()
