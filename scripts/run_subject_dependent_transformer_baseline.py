"""Run a first subject-dependent raw-window Transformer checkpoint on GAMEEMO."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_window_dataset
from src.config import LABEL_MAP, RESULTS_DIR
from src.eeg_transformer import EEGTransformerConfig, build_eeg_transformer
from src.gameemo_loader import GAMEEMO_ROOT
from src.neural_normalization import channel_standardize_splits
from src.splits import make_subject_dependent_split
from src.windowing import DEFAULT_WINDOW_SAMPLES


RANDOM_SEED = 0
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RESULTS_PATH = RESULTS_DIR / "subject_dependent_transformer_baseline.csv"


def transformer_model_name(input_mode: str) -> str:
    if input_mode == "temporal_patch":
        return "TemporalPatchTransformer"
    return "ChannelTokenTransformer"


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
            "PyTorch is required for the Transformer baseline. Install torch before running this script."
        ) from exc
    return torch, DataLoader, TensorDataset


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


def balanced_class_weights(labels, n_classes: int):
    """Compute balanced class weights from training labels only."""
    import numpy as np

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    if (counts == 0).any():
        missing = [str(idx) for idx, count in enumerate(counts) if count == 0]
        raise ValueError(f"Cannot compute balanced class weights; missing train classes: {', '.join(missing)}")
    weights = counts.sum() / (n_classes * counts)
    return weights


def write_result_csv(
    output_path: Path,
    *,
    epochs: int,
    epochs_ran: int,
    best_epoch: int,
    random_seed: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    window_samples: int,
    n_records: int,
    n_windows: int,
    n_channels: int,
    train_windows: int,
    val_windows: int,
    test_windows: int,
    val_accuracy: float,
    val_macro_f1: float,
    accuracy: float,
    macro_f1: float,
    final_train_loss: float,
    best_train_loss: float,
    early_stopped: bool,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    normalization_strategy: str,
    input_mode: str,
    patch_samples: int,
    class_weight: str,
    class_weights,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model_name",
        "random_seed",
        "test_size",
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
        "d_model",
        "n_heads",
        "n_layers",
        "dim_feedforward",
        "input_mode",
        "patch_samples",
        "device",
        "normalization_strategy",
        "class_weight",
        "class_weights",
        "best_val_accuracy",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "final_train_loss",
        "best_train_loss",
        "early_stopped",
        "status",
    ]
    row = {
        "experiment": "subject_dependent_transformer_raw_windows",
        "model_name": transformer_model_name(input_mode),
        "random_seed": random_seed,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "window_samples": window_samples,
        "n_records": n_records,
        "n_windows": n_windows,
        "n_channels": n_channels,
        "n_classes": len(LABEL_MAP),
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "requested_epochs": epochs,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dim_feedforward": dim_feedforward,
        "input_mode": input_mode,
        "patch_samples": patch_samples,
        "device": device,
        "normalization_strategy": normalization_strategy,
        "class_weight": class_weight,
        "class_weights": ";".join(f"{float(weight):.6f}" for weight in class_weights),
        "best_val_accuracy": f"{val_accuracy:.6f}",
        "best_val_macro_f1": f"{val_macro_f1:.6f}",
        "test_accuracy": f"{accuracy:.6f}",
        "test_macro_f1": f"{macro_f1:.6f}",
        "final_train_loss": f"{final_train_loss:.6f}",
        "best_train_loss": f"{best_train_loss:.6f}",
        "early_stopped": early_stopped,
        "status": "first_subject_dependent_checkpoint_not_final",
    }
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a subject-dependent GAMEEMO Transformer checkpoint.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Path to write the result CSV.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--input-mode", choices=["channel", "temporal_patch"], default="channel")
    parser.add_argument("--patch-samples", type=int, default=32)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
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

    if args.epochs <= 0:
        raise SystemExit(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise SystemExit(f"batch-size must be positive, got {args.batch_size}")
    if args.patience <= 0:
        raise SystemExit(f"patience must be positive, got {args.patience}")
    if args.patch_samples <= 0:
        raise SystemExit(f"patch-samples must be positive, got {args.patch_samples}")

    torch, DataLoader, TensorDataset = require_torch()
    import numpy as np
    from sklearn.model_selection import train_test_split

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    dataset = build_window_dataset(
        args.root,
        limit_records=args.limit_records,
        window_samples=args.window_samples,
    )
    split = make_subject_dependent_split(
        dataset.metadata,
        test_size=TEST_SIZE,
        random_state=args.random_seed,
        stratify=True,
    )
    train_indices, val_indices = train_test_split(
        split.train_indices,
        test_size=VAL_SIZE,
        random_state=args.random_seed,
        stratify=dataset.labels[split.train_indices],
    )
    train_indices = sorted(int(idx) for idx in train_indices)
    val_indices = sorted(int(idx) for idx in val_indices)

    x_train, x_val, x_test, normalization_stats = channel_standardize_splits(
        dataset.windows,
        train_indices,
        val_indices,
        split.test_indices,
    )
    y_train = dataset.labels[train_indices]
    y_val = dataset.labels[val_indices]
    y_test = dataset.labels[split.test_indices]

    device = TorchDevice(torch, args.device)
    model = build_eeg_transformer(
        EEGTransformerConfig(
            n_channels=x_train.shape[1],
            n_samples=x_train.shape[2],
            n_classes=len(LABEL_MAP),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            input_mode=args.input_mode,
            patch_samples=args.patch_samples,
        )
    ).to(device.value)

    train_loader = make_loader(torch, DataLoader, TensorDataset, x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(torch, DataLoader, TensorDataset, x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(torch, DataLoader, TensorDataset, x_test, y_test, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.class_weight == "balanced":
        class_weights = balanced_class_weights(y_train, n_classes=len(LABEL_MAP))
        criterion_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device.value)
    else:
        import numpy as np

        class_weights = np.ones(len(LABEL_MAP), dtype=float)
        criterion_weights = None
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weights)
    final_loss = 0.0
    best_train_loss = 0.0
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_macro_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        final_loss = train_one_epoch(model, train_loader, optimizer, criterion, device.value)
        val_accuracy, val_macro_f1 = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={final_loss:.6f} "
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

        if epochs_without_improvement >= args.patience:
            early_stopped = True
            print(f"early_stopping_epoch={epoch}")
            break

    epochs_ran = epoch
    model.load_state_dict(best_state)
    accuracy, macro_f1 = evaluate(model, test_loader, device)

    write_result_csv(
        args.output,
        epochs=args.epochs,
        epochs_ran=epochs_ran,
        best_epoch=best_epoch,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=str(device.value),
        window_samples=args.window_samples,
        n_records=dataset.n_records,
        n_windows=len(dataset.metadata),
        n_channels=dataset.windows.shape[1],
        train_windows=len(train_indices),
        val_windows=len(val_indices),
        test_windows=len(split.test_indices),
        val_accuracy=best_val_accuracy,
        val_macro_f1=best_val_macro_f1,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        final_train_loss=final_loss,
        best_train_loss=best_train_loss,
        early_stopped=early_stopped,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        normalization_strategy=normalization_stats.strategy,
        input_mode=args.input_mode,
        patch_samples=args.patch_samples,
        class_weight=args.class_weight,
        class_weights=class_weights,
    )

    print(f"records: {dataset.n_records}")
    print(f"windows: {dataset.windows.shape}")
    print(f"train_windows: {len(train_indices)}")
    print(f"val_windows: {len(val_indices)}")
    print(f"test_windows: {len(split.test_indices)}")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_accuracy: {best_val_accuracy:.6f}")
    print(f"best_val_macro_f1: {best_val_macro_f1:.6f}")
    print(f"test_accuracy: {accuracy:.6f}")
    print(f"test_macro_f1: {macro_f1:.6f}")
    print(f"normalization_strategy: {normalization_stats.strategy}")
    print(f"input_mode: {args.input_mode}")
    print(f"patch_samples: {args.patch_samples}")
    print(f"class_weight: {args.class_weight}")
    print("class_weights: " + ";".join(f"{float(weight):.6f}" for weight in class_weights))
    print("status: first_subject_dependent_checkpoint_not_final")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
