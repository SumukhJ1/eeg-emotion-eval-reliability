"""Export Transformer LOSO confusion, subject, and class-level analysis.

This reruns the configured LOSO Transformer checkpoint path because the baseline
result files intentionally store fold-level metrics only, not per-window
predictions. The prediction rows saved here contain labels and metadata only, not
raw EEG samples.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_data import build_window_dataset
from src.config import LABEL_MAP, RESULTS_DIR
from src.eeg_transformer import EEGTransformerConfig, build_eeg_transformer
from src.gameemo_loader import GAMEEMO_ROOT
from src.neural_normalization import channel_standardize_splits
from src.splits import MetadataSplit, make_loso_splits, validate_loso_split
from src.windowing import EegWindowMetadata


RANDOM_SEED = 0
VAL_SIZE = 0.2
PREDICTION_PATH = RESULTS_DIR / "transformer_loso_predictions.csv"
CONFUSION_PATH = RESULTS_DIR / "transformer_loso_confusion_matrix.csv"
SUBJECT_ANALYSIS_PATH = RESULTS_DIR / "transformer_loso_subject_analysis.csv"
CLASS_F1_PATH = RESULTS_DIR / "transformer_loso_class_f1.csv"

INDEX_TO_LABEL = {index: label for label, index in LABEL_MAP.items()}
ORDERED_LABELS = [label for label, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]


@dataclass(frozen=True)
class FoldMetrics:
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
    normalization_strategy: str


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
        raise SystemExit("PyTorch is required for Transformer LOSO analysis.") from exc
    return torch, DataLoader, TensorDataset


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
        raise ValueError(f"LOSO leakage: held-out subject {heldout} appears in training")
    if heldout in val_subjects:
        raise ValueError(f"LOSO leakage: held-out subject {heldout} appears in validation")


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


def evaluate_with_predictions(model, loader, device) -> tuple[float, float, list[int], list[int]]:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with device.no_grad_context():
        for inputs, batch_targets in loader:
            logits = model(inputs.to(device.value))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(batch_targets.tolist())
    return (
        float(accuracy_score(targets, predictions)),
        float(f1_score(targets, predictions, average="macro")),
        predictions,
        targets,
    )


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
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    input_mode: str,
    patch_samples: int,
    random_seed: int,
) -> tuple[FoldMetrics, list[int], list[int]]:
    train_indices, val_indices = split_train_validation(
        split.train_indices,
        labels,
        val_size=val_size,
        random_seed=random_seed,
    )
    assert_no_subject_leakage(metadata, split, train_indices, val_indices)

    x_train, x_val, x_test, normalization_stats = channel_standardize_splits(
        windows,
        train_indices,
        val_indices,
        split.test_indices,
    )
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[split.test_indices]

    model = build_eeg_transformer(
        EEGTransformerConfig(
            n_channels=x_train.shape[1],
            n_samples=x_train.shape[2],
            n_classes=len(LABEL_MAP),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            input_mode=input_mode,
            patch_samples=patch_samples,
        )
    ).to(device.value)
    train_loader = make_loader(torch, DataLoader, TensorDataset, x_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(torch, DataLoader, TensorDataset, x_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(torch, DataLoader, TensorDataset, x_test, y_test, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        val_accuracy, val_macro_f1, _, _ = evaluate_with_predictions(model, val_loader, device)
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
    test_accuracy, test_macro_f1, predictions, targets = evaluate_with_predictions(model, test_loader, device)
    return (
        FoldMetrics(
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
            normalization_strategy=normalization_stats.strategy,
        ),
        predictions,
        targets,
    )


def prediction_fieldnames() -> list[str]:
    return [
        "experiment",
        "model_name",
        "protocol",
        "subject",
        "game",
        "random_seed",
        "window_samples",
        "input_mode",
        "patch_samples",
        "true_label",
        "predicted_label",
        "correct",
        "source_file",
        "start_sample",
        "end_sample",
    ]


def write_prediction_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=prediction_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    if reader.fieldnames != prediction_fieldnames():
        raise ValueError(f"Existing prediction file has unexpected columns: {path}")
    return rows


def prediction_rows_for_fold(
    *,
    dataset_metadata: list[EegWindowMetadata],
    test_indices: list[int],
    predictions: list[int],
    targets: list[int],
    random_seed: int,
    window_samples: int,
    input_mode: str,
    patch_samples: int,
) -> list[dict[str, object]]:
    if len(test_indices) != len(predictions) or len(predictions) != len(targets):
        raise ValueError("Prediction, target, and metadata index counts do not match")
    rows: list[dict[str, object]] = []
    for idx, predicted_idx, target_idx in zip(test_indices, predictions, targets):
        meta = dataset_metadata[idx]
        true_label = INDEX_TO_LABEL[int(target_idx)]
        predicted_label = INDEX_TO_LABEL[int(predicted_idx)]
        rows.append(
            {
                "experiment": "transformer_loso_prediction_analysis",
                "model_name": "TemporalPatchTransformer" if input_mode == "temporal_patch" else "ChannelTokenTransformer",
                "protocol": "LOSO",
                "subject": meta.subject,
                "game": meta.game,
                "random_seed": random_seed,
                "window_samples": window_samples,
                "input_mode": input_mode,
                "patch_samples": patch_samples,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
                "source_file": str(meta.source_file),
                "start_sample": meta.start_sample,
                "end_sample": meta.end_sample,
            }
        )
    return rows


def write_confusion_matrix(prediction_rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "scope",
        "heldout_subject",
        "true_label",
        "predicted_label",
        "count",
        "normalized_by_true",
    ]
    subjects = sorted({str(row["subject"]) for row in prediction_rows})
    output_rows: list[dict[str, object]] = []

    for subject in ["ALL", *subjects]:
        subset = prediction_rows if subject == "ALL" else [row for row in prediction_rows if row["subject"] == subject]
        counts = Counter((str(row["true_label"]), str(row["predicted_label"])) for row in subset)
        true_totals = Counter(str(row["true_label"]) for row in subset)
        for true_label in ORDERED_LABELS:
            for predicted_label in ORDERED_LABELS:
                count = counts[(true_label, predicted_label)]
                total = true_totals[true_label]
                output_rows.append(
                    {
                        "scope": "aggregate" if subject == "ALL" else "heldout_subject",
                        "heldout_subject": subject,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                        "normalized_by_true": f"{(count / total) if total else 0.0:.6f}",
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def macro_f1_for_rows(rows: list[dict[str, object]]) -> float:
    from sklearn.metrics import f1_score

    targets = [LABEL_MAP[str(row["true_label"])] for row in rows]
    predictions = [LABEL_MAP[str(row["predicted_label"])] for row in rows]
    return float(f1_score(targets, predictions, labels=list(range(len(LABEL_MAP))), average="macro"))


def write_subject_analysis(prediction_rows: list[dict[str, object]], path: Path) -> None:
    by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_subject[str(row["subject"])].append(row)

    metrics = []
    for subject, rows in sorted(by_subject.items()):
        correct = sum(str(row["correct"]) == "True" or row["correct"] is True for row in rows)
        accuracy = correct / len(rows)
        metrics.append(
            {
                "heldout_subject": subject,
                "test_windows": len(rows),
                "accuracy": accuracy,
                "macro_f1": macro_f1_for_rows(rows),
            }
        )

    ranked_by_f1 = sorted(metrics, key=lambda row: row["macro_f1"], reverse=True)
    best_subject = ranked_by_f1[0]["heldout_subject"] if ranked_by_f1 else ""
    worst_subject = ranked_by_f1[-1]["heldout_subject"] if ranked_by_f1 else ""
    rank_lookup = {row["heldout_subject"]: rank for rank, row in enumerate(ranked_by_f1, start=1)}

    fieldnames = [
        "heldout_subject",
        "test_windows",
        "accuracy",
        "macro_f1",
        "rank_by_macro_f1",
        "best_heldout_subject",
        "worst_heldout_subject",
    ]
    output_rows = [
        {
            "heldout_subject": row["heldout_subject"],
            "test_windows": row["test_windows"],
            "accuracy": f"{row['accuracy']:.6f}",
            "macro_f1": f"{row['macro_f1']:.6f}",
            "rank_by_macro_f1": rank_lookup[row["heldout_subject"]],
            "best_heldout_subject": best_subject,
            "worst_heldout_subject": worst_subject,
        }
        for row in sorted(metrics, key=lambda row: row["heldout_subject"])
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_class_f1(prediction_rows: list[dict[str, object]], path: Path) -> None:
    from sklearn.metrics import precision_recall_fscore_support

    targets = [LABEL_MAP[str(row["true_label"])] for row in prediction_rows]
    predictions = [LABEL_MAP[str(row["predicted_label"])] for row in prediction_rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(ORDERED_LABELS))),
        zero_division=0,
    )
    fieldnames = ["label", "precision", "recall", "f1", "support"]
    output_rows = [
        {
            "label": label,
            "precision": f"{precision[idx]:.6f}",
            "recall": f"{recall[idx]:.6f}",
            "f1": f"{f1[idx]:.6f}",
            "support": int(support[idx]),
        }
        for idx, label in enumerate(ORDERED_LABELS)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_analysis_outputs(prediction_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    write_confusion_matrix(prediction_rows, args.confusion_output)
    write_subject_analysis(prediction_rows, args.subject_output)
    write_class_f1(prediction_rows, args.class_f1_output)
    print(f"wrote: {args.confusion_output}")
    print(f"wrote: {args.subject_output}")
    print(f"wrote: {args.class_f1_output}")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit(f"epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise SystemExit(f"batch-size must be positive, got {args.batch_size}")
    if args.patience <= 0:
        raise SystemExit(f"patience must be positive, got {args.patience}")
    if not 0 < args.val_size < 1:
        raise SystemExit(f"val-size must be between 0 and 1, got {args.val_size}")
    if args.patch_samples <= 0:
        raise SystemExit(f"patch-samples must be positive, got {args.patch_samples}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GAMEEMO Transformer LOSO predictions.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT)
    parser.add_argument("--prediction-output", type=Path, default=PREDICTION_PATH)
    parser.add_argument("--confusion-output", type=Path, default=CONFUSION_PATH)
    parser.add_argument("--subject-output", type=Path, default=SUBJECT_ANALYSIS_PATH)
    parser.add_argument("--class-f1-output", type=Path, default=CLASS_F1_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
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
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--window-samples", type=int, default=512)
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()
    validate_args(args)

    torch, DataLoader, TensorDataset = require_torch()
    import numpy as np

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

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
    prediction_rows: list[dict[str, object]] = (
        list(read_prediction_rows(args.prediction_output)) if args.resume_existing else []
    )
    completed_subjects = {str(row["subject"]) for row in prediction_rows}
    splits = [split for split in splits if split.test_subject not in completed_subjects]

    print(f"records: {dataset.n_records}")
    print(f"windows: {dataset.windows.shape}")
    print(f"folds_to_run: {len(splits)}")
    print(f"device: {device.value}")
    if completed_subjects:
        print(f"resumed_completed_subjects: {','.join(sorted(completed_subjects))}")

    if args.resume_existing and not splits:
        write_analysis_outputs(prediction_rows, args)
        return

    for split in splits:
        fold_metrics, predictions, targets = run_fold(
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
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            input_mode=args.input_mode,
            patch_samples=args.patch_samples,
            random_seed=args.random_seed,
        )
        prediction_rows.extend(
            prediction_rows_for_fold(
                dataset_metadata=dataset.metadata,
                test_indices=split.test_indices,
                predictions=predictions,
                targets=targets,
                random_seed=args.random_seed,
                window_samples=args.window_samples,
                input_mode=args.input_mode,
                patch_samples=args.patch_samples,
            )
        )
        write_prediction_rows(args.prediction_output, prediction_rows)
        write_analysis_outputs(prediction_rows, args)
        print(
            f"subject={fold_metrics.subject} test_accuracy={fold_metrics.test_accuracy:.6f} "
            f"test_macro_f1={fold_metrics.test_macro_f1:.6f} best_epoch={fold_metrics.best_epoch}"
        )

    print(f"wrote: {args.prediction_output}")


if __name__ == "__main__":
    main()
