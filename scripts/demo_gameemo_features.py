"""Extract baseline statistical features from one GAMEEMO recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_statistical_features
from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.windowing import DEFAULT_WINDOW_SAMPLES, window_gameemo_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo GAMEEMO window-level statistical features.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument(
        "--window-samples",
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help="Fixed window length in samples. Defaults to 2 seconds at 128 Hz.",
    )
    args = parser.parse_args()

    records = discover_records(args.root, limit=1)
    if not records:
        raise SystemExit(f"No preprocessed GAMEEMO CSV records found under {args.root}")

    record, data = load_record(records[0])
    windows, _ = window_gameemo_record(record, data, window_samples=args.window_samples)
    features, feature_names = extract_statistical_features(windows, channels=record.channels)

    print(f"source_file: {record.source_file}")
    print(f"record: subject={record.subject} game={record.game} label={record.label}")
    print(f"window_shape: {windows.shape}")
    print(f"feature_matrix_shape: {features.shape}")
    print(f"n_feature_names: {len(feature_names)}")
    print(f"first_feature_names: {feature_names[:10]}")


if __name__ == "__main__":
    main()
