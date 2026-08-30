"""Audit basic signal quality flags for GAMEEMO preprocessed EEG CSV windows."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.windowing import DEFAULT_WINDOW_SAMPLES, window_gameemo_record


SUMMARY_OUTPUT = RESULTS_DIR / "gameemo_signal_quality_audit_summary.csv"
SUBJECT_OUTPUT = RESULTS_DIR / "gameemo_signal_quality_audit_by_subject.csv"
FLAGGED_WINDOWS_OUTPUT = RESULTS_DIR / "gameemo_signal_quality_flagged_windows.csv"


@dataclass(frozen=True)
class WindowQualityRow:
    subject: str
    game: str
    label: str
    source_file: Path
    start_sample: int
    end_sample: int
    has_missing: bool
    has_nonfinite: bool
    near_zero_variance: bool
    flat_channel_count: int
    max_abs_amplitude: float
    window_variance: float


def robust_upper_threshold(values, multiplier: float) -> float:
    import numpy as np

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return math.inf

    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    if mad == 0.0:
        q1, q3 = np.percentile(finite_values, [25, 75])
        iqr = float(q3 - q1)
        if iqr == 0.0:
            return float(finite_values.max())
        return float(q3 + multiplier * iqr)
    return median + multiplier * mad


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def build_quality_rows(root: Path, window_samples: int, variance_epsilon: float) -> list[WindowQualityRow]:
    import numpy as np

    rows: list[WindowQualityRow] = []
    records = discover_records(root)
    if not records:
        raise ValueError(f"No GAMEEMO preprocessed CSV records found under {root}")

    for record in records:
        loaded_record, data = load_record(record)
        windows, metadata = window_gameemo_record(
            loaded_record,
            data,
            window_samples=window_samples,
        )
        if windows.shape[0] == 0:
            continue

        missing_mask = np.isnan(windows)
        nonfinite_mask = ~np.isfinite(windows)
        channel_variance = np.nanvar(windows, axis=2)
        window_variance = np.nanvar(windows, axis=(1, 2))
        max_abs = np.nanmax(np.abs(windows), axis=(1, 2))

        flat_channel_counts = (channel_variance <= variance_epsilon).sum(axis=1)
        near_zero_windows = window_variance <= variance_epsilon

        for window_idx, window_metadata in enumerate(metadata):
            rows.append(
                WindowQualityRow(
                    subject=window_metadata.subject,
                    game=window_metadata.game,
                    label=window_metadata.label,
                    source_file=window_metadata.source_file,
                    start_sample=window_metadata.start_sample,
                    end_sample=window_metadata.end_sample,
                    has_missing=bool(missing_mask[window_idx].any()),
                    has_nonfinite=bool(nonfinite_mask[window_idx].any()),
                    near_zero_variance=bool(near_zero_windows[window_idx]),
                    flat_channel_count=int(flat_channel_counts[window_idx]),
                    max_abs_amplitude=float(max_abs[window_idx]),
                    window_variance=float(window_variance[window_idx]),
                )
            )

    return rows


def summarize(rows: list[WindowQualityRow], amplitude_threshold: float, variance_threshold: float) -> dict[str, object]:
    subjects = {row.subject for row in rows}
    source_files = {row.source_file for row in rows}
    missing_windows = [row for row in rows if row.has_missing]
    nonfinite_windows = [row for row in rows if row.has_nonfinite]
    near_zero_windows = [row for row in rows if row.near_zero_variance]
    flat_channel_windows = [row for row in rows if row.flat_channel_count > 0]
    extreme_amplitude_windows = [row for row in rows if row.max_abs_amplitude > amplitude_threshold]
    extreme_variance_windows = [row for row in rows if row.window_variance > variance_threshold]

    return {
        "n_subjects": len(subjects),
        "n_records": len(source_files),
        "n_windows": len(rows),
        "missing_value_windows": len(missing_windows),
        "nonfinite_value_windows": len(nonfinite_windows),
        "near_zero_variance_windows": len(near_zero_windows),
        "flat_channel_windows": len(flat_channel_windows),
        "flat_channel_instances": sum(row.flat_channel_count for row in rows),
        "extreme_amplitude_windows": len(extreme_amplitude_windows),
        "extreme_variance_windows": len(extreme_variance_windows),
        "amplitude_threshold": format_float(amplitude_threshold),
        "variance_threshold": format_float(variance_threshold),
    }


def summarize_by_subject(
    rows: list[WindowQualityRow],
    amplitude_threshold: float,
    variance_threshold: float,
) -> list[dict[str, object]]:
    grouped: dict[str, list[WindowQualityRow]] = defaultdict(list)
    for row in rows:
        grouped[row.subject].append(row)

    subject_rows = []
    for subject, subject_windows in sorted(grouped.items()):
        subject_rows.append(
            {
                "subject": subject,
                "n_windows": len(subject_windows),
                "missing_value_windows": sum(row.has_missing for row in subject_windows),
                "nonfinite_value_windows": sum(row.has_nonfinite for row in subject_windows),
                "near_zero_variance_windows": sum(row.near_zero_variance for row in subject_windows),
                "flat_channel_windows": sum(row.flat_channel_count > 0 for row in subject_windows),
                "flat_channel_instances": sum(row.flat_channel_count for row in subject_windows),
                "extreme_amplitude_windows": sum(
                    row.max_abs_amplitude > amplitude_threshold for row in subject_windows
                ),
                "extreme_variance_windows": sum(row.window_variance > variance_threshold for row in subject_windows),
            }
        )
    return subject_rows


def flagged_window_rows(
    rows: list[WindowQualityRow],
    amplitude_threshold: float,
    variance_threshold: float,
) -> list[dict[str, object]]:
    flagged = []
    for row in rows:
        extreme_amplitude = row.max_abs_amplitude > amplitude_threshold
        extreme_variance = row.window_variance > variance_threshold
        if not (
            row.has_missing
            or row.has_nonfinite
            or row.near_zero_variance
            or row.flat_channel_count > 0
            or extreme_amplitude
            or extreme_variance
        ):
            continue
        flagged.append(
            {
                "subject": row.subject,
                "game": row.game,
                "label": row.label,
                "source_file": row.source_file,
                "start_sample": row.start_sample,
                "end_sample": row.end_sample,
                "has_missing": row.has_missing,
                "has_nonfinite": row.has_nonfinite,
                "near_zero_variance": row.near_zero_variance,
                "flat_channel_count": row.flat_channel_count,
                "extreme_amplitude": extreme_amplitude,
                "extreme_variance": extreme_variance,
                "max_abs_amplitude": format_float(row.max_abs_amplitude),
                "window_variance": format_float(row.window_variance),
            }
        )
    return flagged


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit basic GAMEEMO signal quality flags.")
    parser.add_argument("--root", type=Path, default=GAMEEMO_ROOT, help="Path to the GAMEEMO dataset root.")
    parser.add_argument("--window-samples", type=int, default=DEFAULT_WINDOW_SAMPLES)
    parser.add_argument(
        "--variance-epsilon",
        type=float,
        default=1e-12,
        help="Variance at or below this value is treated as flat or near-zero.",
    )
    parser.add_argument(
        "--robust-multiplier",
        type=float,
        default=10.0,
        help="Multiplier for median absolute deviation thresholds for extreme window flags.",
    )
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument("--subject-output", type=Path, default=SUBJECT_OUTPUT)
    parser.add_argument("--flagged-output", type=Path, default=FLAGGED_WINDOWS_OUTPUT)
    args = parser.parse_args()

    if args.window_samples <= 0:
        raise SystemExit(f"window-samples must be positive, got {args.window_samples}")
    if args.variance_epsilon < 0:
        raise SystemExit(f"variance-epsilon must be non-negative, got {args.variance_epsilon}")
    if args.robust_multiplier <= 0:
        raise SystemExit(f"robust-multiplier must be positive, got {args.robust_multiplier}")

    rows = build_quality_rows(args.root, args.window_samples, args.variance_epsilon)
    if not rows:
        raise SystemExit("No GAMEEMO windows available for quality audit.")

    import numpy as np

    amplitude_values = np.asarray([row.max_abs_amplitude for row in rows], dtype=np.float64)
    variance_values = np.asarray([row.window_variance for row in rows], dtype=np.float64)
    amplitude_threshold = robust_upper_threshold(amplitude_values, args.robust_multiplier)
    variance_threshold = robust_upper_threshold(variance_values, args.robust_multiplier)

    summary_row = summarize(rows, amplitude_threshold, variance_threshold)
    subject_rows = summarize_by_subject(rows, amplitude_threshold, variance_threshold)
    flagged_rows = flagged_window_rows(rows, amplitude_threshold, variance_threshold)

    write_csv(args.summary_output, [summary_row])
    write_csv(args.subject_output, subject_rows)
    write_csv(args.flagged_output, flagged_rows)

    print(f"n_subjects: {summary_row['n_subjects']}")
    print(f"n_records: {summary_row['n_records']}")
    print(f"n_windows: {summary_row['n_windows']}")
    print(f"missing_value_windows: {summary_row['missing_value_windows']}")
    print(f"nonfinite_value_windows: {summary_row['nonfinite_value_windows']}")
    print(f"near_zero_variance_windows: {summary_row['near_zero_variance_windows']}")
    print(f"flat_channel_windows: {summary_row['flat_channel_windows']}")
    print(f"flat_channel_instances: {summary_row['flat_channel_instances']}")
    print(f"extreme_amplitude_windows: {summary_row['extreme_amplitude_windows']}")
    print(f"extreme_variance_windows: {summary_row['extreme_variance_windows']}")
    print(f"amplitude_threshold: {summary_row['amplitude_threshold']}")
    print(f"variance_threshold: {summary_row['variance_threshold']}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.subject_output}")
    print(f"wrote: {args.flagged_output}")


if __name__ == "__main__":
    main()
