"""Inspect the local GAMEEMO dataset layout without copying any data files."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


GAMEEMO_ROOT = Path(
    r"C:\Users\Sumukh\Downloads\GAMEEMO Dataset"
    r"\Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO"
    r"\GAMEEMO"
)
SAMPLING_RATE_HZ = 128
COUNT_SUFFIXES = {".csv", ".mat", ".pdf", ".mp4"}


def is_subject_dir(path: Path) -> bool:
    name = path.name.strip()
    return path.is_dir() and name.startswith("(S") and name.endswith(")")


def count_files(paths: list[Path]) -> Counter[str]:
    return Counter(path.suffix.lower() for path in paths if path.suffix.lower() in COUNT_SUFFIXES)


def print_counts(title: str, files: list[Path]) -> None:
    counts = count_files(files)
    print(f"{title}:")
    for suffix in sorted(COUNT_SUFFIXES):
        print(f"  {suffix}: {counts.get(suffix, 0)}")


def find_eeg_files(subject_dirs: list[Path], data_folder_name: str) -> list[Path]:
    files: list[Path] = []
    for subject_dir in subject_dirs:
        data_dir = subject_dir / data_folder_name
        if data_dir.exists():
            files.extend(path for path in data_dir.rglob("*") if path.is_file())
    return sorted(files)


def trim_trailing_empty_fields(row: list[str]) -> list[str]:
    while row and row[-1] == "":
        row = row[:-1]
    return row


def inspect_csv(csv_path: Path) -> tuple[list[str], int, int]:
    """Return header, data row count, and column count using the standard library."""
    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = trim_trailing_empty_fields(next(reader, []))
        rows = 0
        max_columns = len(header)
        for row in reader:
            row = trim_trailing_empty_fields(row)
            if not row:
                continue
            rows += 1
            max_columns = max(max_columns, len(row))

    return header, rows, max_columns


def main() -> None:
    root = GAMEEMO_ROOT
    if not root.exists():
        raise SystemExit(f"GAMEEMO root not found: {root}")

    subject_dirs = sorted(path for path in root.iterdir() if is_subject_dir(path))
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    raw_files = find_eeg_files(subject_dirs, "Raw EEG Data")
    preprocessed_files = find_eeg_files(subject_dirs, "Preprocessed EEG Data")

    print(f"GAMEEMO root: {root}")
    print(f"Subject folders: {len(subject_dirs)}")
    print_counts("Raw EEG files", raw_files)
    print_counts("Preprocessed EEG files", preprocessed_files)
    print_counts("All dataset files", all_files)

    eeg_samples = sorted(
        path
        for path in raw_files + preprocessed_files
        if path.suffix.lower() in {".csv", ".mat"}
    )
    print("Sample EEG file paths:")
    for path in eeg_samples[:6]:
        print(f"  {path}")

    preprocessed_csvs = [path for path in preprocessed_files if path.suffix.lower() == ".csv"]
    if not preprocessed_csvs:
        print("No preprocessed CSV files found to inspect.")
        return

    csv_path = preprocessed_csvs[0]
    header, rows, columns = inspect_csv(csv_path)
    duration_seconds = rows / SAMPLING_RATE_HZ

    print("Preprocessed CSV inspection:")
    print(f"  path: {csv_path}")
    print(f"  header: {header}")
    print(f"  shape: rows={rows}, columns={columns}")
    print(f"  duration_seconds_at_{SAMPLING_RATE_HZ}_hz: {duration_seconds:.2f}")


if __name__ == "__main__":
    main()
