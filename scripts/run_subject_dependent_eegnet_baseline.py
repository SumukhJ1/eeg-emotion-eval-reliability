"""Run a first subject-dependent EEGNet baseline on GAMEEMO windows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_window_dataset
from src.config import LABEL_MAP, RESULTS_DIR
from src.eegnet import EEGNetConfig, build_eegnet
from src.gameemo_loader import GAMEEMO_ROOT
from src.splits import make_subject_dependent_split
from src.windowing import DEFAULT_WINDOW_SAMPLES


RANDOM_SEED = 0
TEST_SIZE = 0.2
RESULTS_PATH = RESULTS_DIR / "subject_dependent_eegnet_baseline.csv"


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the EEGNet baseline. Install torch before running this script."
        ) from exc
    return torch, DataLoader, TensorDataset


def channel_standardize(windows, train_indices, test_indices):
    train_block = windows[train_indices]
    mean = train_block.mean(axis=(0, 2), keepdims=True)
    std = train_block.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return (windows[train_indices] - mean) / std, (windows[test_indices] - mean) / std


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


def predict(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with device.no_grad_context():
        for inputs, batch_targets in loader:
            logits = model(inputs.to(device.value))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(batch_targets.tolist())
    return targets, predictions


class TorchDevice:
    def __init__(self, torch_module, requested: str) -> None:
        if requested == "auto":
            requested = "cuda" if torch_module.cuda.is_available() else "cpu"
        self.value = torch_module.device(requested)
        self._torch = torch_module

    def no_grad_context(self):
        return self._torch.no_grad()


def write_result_csv(
    output_path: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    window_samples: int,
    n_records: int,
    n_windows: int,
    n_channels: int,
    train_windows: int,
    test_windows: int,
    accuracy: float,
    macro_f1: float,
    final_train_loss: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model_name",
        "random_seed",
        "test_size",
        "window_samples",
        "n_records",
        "n_windows",
        "n_channels",
        "n_classes",
        "train_windows",
        "test_windows",
        "epochs",
        "batch_size",
        "learning_rate",
        "device",
        "accuracy",
        "macro_f1",
        "final_train_loss",
    ]
    row = {
        "experiment": "subject_dependent_eegnet_raw_windows",
        "model_name": "EEGNet",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "window_samples": window_samples,
        "n_records": n_records,
        "n_windows": n_windows,
        "n_channels": n_channels,
        "n_classes": len(LABEL_MAP),
        "train_windows": train_windows,
        "test_windows": test_windows,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "accuracy": f"{accuracy:.6f}",
        "macro_f1": f"{macro_f1:.6f}",
        "final_train_loss": f"{final_train_loss:.6f}",
    }
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a subject-dependent GAMEEMO EEGNet baseline.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Path to write the result CSV.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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

    torch, DataLoader, TensorDataset = require_torch()
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset = build_window_dataset(
        args.root,
        limit_records=args.limit_records,
        window_samples=args.window_samples,
    )
    split = make_subject_dependent_split(
        dataset.metadata,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=True,
    )
    x_train, x_test = channel_standardize(dataset.windows, split.train_indices, split.test_indices)
    y_train = dataset.labels[split.train_indices]
    y_test = dataset.labels[split.test_indices]

    device = TorchDevice(torch, args.device)
    model = build_eegnet(
        EEGNetConfig(
            n_channels=x_train.shape[1],
            n_samples=x_train.shape[2],
            n_classes=len(LABEL_MAP),
        )
    ).to(device.value)

    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_test, dtype=torch.float32),
            torch.as_tensor(y_test, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        final_loss = train_one_epoch(model, train_loader, optimizer, criterion, device.value)
        print(f"epoch={epoch} train_loss={final_loss:.6f}")

    targets, predictions = predict(model, test_loader, device)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro")

    write_result_csv(
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=str(device.value),
        window_samples=args.window_samples,
        n_records=dataset.n_records,
        n_windows=len(dataset.metadata),
        n_channels=dataset.windows.shape[1],
        train_windows=len(split.train_indices),
        test_windows=len(split.test_indices),
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        final_train_loss=final_loss,
    )

    print(f"records: {dataset.n_records}")
    print(f"windows: {dataset.windows.shape}")
    print(f"train_windows: {len(split.train_indices)}")
    print(f"test_windows: {len(split.test_indices)}")
    print(f"accuracy: {accuracy:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
