"""Preview raw-window Transformer input/output shapes on one GAMEEMO recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_MAP
from src.eeg_transformer import EEGTransformerConfig, build_eeg_transformer
from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.windowing import DEFAULT_WINDOW_SAMPLES, window_gameemo_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a raw-window Transformer shape check on one GAMEEMO CSV.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of windows to use for the shape check.")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--window-samples",
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help="Fixed window length in samples. Defaults to 2 seconds at 128 Hz.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit(f"batch-size must be positive, got {args.batch_size}")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed, so the Transformer forward-pass demo cannot run. "
            "Install torch before training deep-learning models."
        ) from exc

    records = discover_records(args.root, limit=1)
    if not records:
        raise SystemExit(f"No preprocessed GAMEEMO CSV records found under {args.root}")

    record, data = load_record(records[0])
    windows, _ = window_gameemo_record(record, data, window_samples=args.window_samples)
    if windows.shape[0] == 0:
        raise SystemExit(f"Recording is shorter than one full {args.window_samples}-sample window: {record.source_file}")

    batch = torch.as_tensor(windows[: args.batch_size], dtype=torch.float32)
    config = EEGTransformerConfig(
        n_channels=batch.shape[1],
        n_samples=batch.shape[2],
        n_classes=len(LABEL_MAP),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = build_eeg_transformer(config)
    model.eval()

    with torch.no_grad():
        logits = model(batch)

    print(f"source_file: {record.source_file}")
    print(f"record: subject={record.subject} game={record.game} label={record.label}")
    print(f"window_shape: {windows.shape}")
    print(f"batch_shape: {tuple(batch.shape)}")
    print(f"token_count: {config.n_channels + 1}")
    print(f"channel_tokens: {config.n_channels}")
    print(f"cls_tokens: 1")
    print(f"d_model: {config.d_model}")
    print(f"n_heads: {config.n_heads}")
    print(f"n_layers: {config.n_layers}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"n_classes: {config.n_classes}")
    print("training_performed: False")


if __name__ == "__main__":
    main()
