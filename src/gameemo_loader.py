"""Discover and load preprocessed GAMEEMO CSV EEG recordings.

GAMEEMO game-condition label mapping:

- G1: boring, recorded during Train Sim World
- G2: calm, recorded during Unravel
- G3: horror, recorded during Slender - The Arrival
- G4: funny, recorded during Goat Simulator
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


GAMEEMO_ROOT = Path(
    r"C:\Users\Sumukh\Downloads\GAMEEMO Dataset"
    r"\Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO"
    r"\GAMEEMO"
)

GAME_LABELS = {
    "G1": "boring",
    "G2": "calm",
    "G3": "horror",
    "G4": "funny",
}

FILENAME_PATTERN = re.compile(r"(?P<subject>S\d{2})(?P<game>G\d)AllChannels\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class GameemoCsvRecord:
    subject: str
    game: str
    label: str
    source_file: Path
    n_samples: int
    n_channels: int
    channels: tuple[str, ...]


def trim_trailing_empty_fields(row: list[str]) -> list[str]:
    while row and row[-1] == "":
        row = row[:-1]
    return row


def parse_subject_game(path: Path) -> tuple[str, str]:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Could not parse subject/game IDs from filename: {path.name}")

    subject = match.group("subject").upper()
    game = match.group("game").upper()
    if game not in GAME_LABELS:
        raise ValueError(f"Unsupported GAMEEMO game ID {game} in {path}")

    return subject, game


def preprocessed_csv_dir(subject_dir: Path) -> Path:
    return subject_dir / "Preprocessed EEG Data" / ".csv format"


def discover_preprocessed_csvs(root: Path = GAMEEMO_ROOT) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"GAMEEMO root not found: {root}")

    csv_files: list[Path] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("(S"):
            continue
        csv_dir = preprocessed_csv_dir(subject_dir)
        if csv_dir.exists():
            csv_files.extend(sorted(csv_dir.glob("*AllChannels.csv")))

    return sorted(csv_files)


def inspect_csv_metadata(path: Path) -> tuple[tuple[str, ...], int, int]:
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        channels = tuple(trim_trailing_empty_fields(next(reader, [])))
        n_samples = 0
        n_channels = len(channels)

        for row in reader:
            row = trim_trailing_empty_fields(row)
            if not row:
                continue
            n_samples += 1
            n_channels = max(n_channels, len(row))

    return channels, n_samples, n_channels


def build_record(path: Path) -> GameemoCsvRecord:
    subject, game = parse_subject_game(path)
    channels, n_samples, n_channels = inspect_csv_metadata(path)
    return GameemoCsvRecord(
        subject=subject,
        game=game,
        label=GAME_LABELS[game],
        source_file=path,
        n_samples=n_samples,
        n_channels=n_channels,
        channels=channels,
    )


def discover_records(root: Path = GAMEEMO_ROOT, limit: int | None = None) -> list[GameemoCsvRecord]:
    csv_files = discover_preprocessed_csvs(root)
    if limit is not None:
        csv_files = csv_files[:limit]
    return [build_record(path) for path in csv_files]


def load_csv_array(path: Path) -> "np.ndarray":
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required to load GAMEEMO CSV data arrays.") from exc

    rows: list[list[float]] = []
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            row = trim_trailing_empty_fields(row)
            if not row:
                continue
            rows.append([float(value) for value in row])

    return np.asarray(rows, dtype=np.float64)


def load_record(record: GameemoCsvRecord) -> tuple[GameemoCsvRecord, "np.ndarray"]:
    data = load_csv_array(record.source_file)
    if data.shape != (record.n_samples, record.n_channels):
        raise ValueError(
            f"Loaded shape {data.shape} does not match metadata "
            f"({record.n_samples}, {record.n_channels}) for {record.source_file}"
        )
    return record, data


def load_dataset(root: Path = GAMEEMO_ROOT) -> list[tuple[GameemoCsvRecord, "np.ndarray"]]:
    return [load_record(record) for record in discover_records(root)]


def print_record(record: GameemoCsvRecord) -> None:
    print(
        f"{record.subject} {record.game} label={record.label} "
        f"shape=({record.n_samples}, {record.n_channels}) file={record.source_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview GAMEEMO preprocessed CSV records.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--limit", type=int, default=5, help="Number of records to preview.")
    parser.add_argument(
        "--load-first",
        action="store_true",
        help="Load the first previewed CSV into a numpy array and print its dtype/shape.",
    )
    args = parser.parse_args()

    records = discover_records(args.root, limit=args.limit)
    print(f"Discovered preview records: {len(records)}")
    for record in records:
        print_record(record)

    if args.load_first and records:
        try:
            record, data = load_record(records[0])
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Loaded first record: {record.subject} {record.game} dtype={data.dtype} shape={data.shape}")


if __name__ == "__main__":
    main()
