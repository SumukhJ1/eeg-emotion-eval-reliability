"""Load one GAMEEMO preprocessed CSV and print its fixed-window shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.windowing import DEFAULT_WINDOW_SAMPLES, window_gameemo_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the window shape for one GAMEEMO CSV recording.")
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
    windows, metadata = window_gameemo_record(record, data, window_samples=args.window_samples)

    print(f"source_file: {record.source_file}")
    print(f"record: subject={record.subject} game={record.game} label={record.label}")
    print(f"recording_shape: {data.shape}")
    print(f"window_shape: {windows.shape}")
    if metadata:
        first = metadata[0]
        print(f"first_window: start_sample={first.start_sample} end_sample={first.end_sample}")
        last = metadata[-1]
        print(f"last_window: start_sample={last.start_sample} end_sample={last.end_sample}")


if __name__ == "__main__":
    main()
