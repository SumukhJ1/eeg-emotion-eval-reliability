"""Fixed-length EEG windowing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import SAMPLING_RATE, WINDOW_SECONDS
from src.gameemo_loader import GameemoCsvRecord

if TYPE_CHECKING:
    import numpy as np


DEFAULT_WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS


@dataclass(frozen=True)
class EegWindowMetadata:
    subject: str
    game: str
    label: str
    source_file: Path
    start_sample: int
    end_sample: int


def window_eeg_array(data: "np.ndarray", window_samples: int = DEFAULT_WINDOW_SAMPLES) -> "np.ndarray":
    """Convert time x channels EEG data into windows shaped windows x channels x samples.

    Leftover samples shorter than a full window are dropped. `end_sample` values
    in companion metadata are exclusive, matching Python slicing.
    """
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}")
    if data.ndim != 2:
        raise ValueError(f"Expected data shaped time_samples x channels, got {data.shape}")

    n_windows = data.shape[0] // window_samples
    usable_samples = n_windows * window_samples
    trimmed = data[:usable_samples]
    if n_windows == 0:
        return data[:0].T.reshape(0, data.shape[1], window_samples)

    return trimmed.reshape(n_windows, window_samples, data.shape[1]).transpose(0, 2, 1)


def build_window_metadata(
    record: GameemoCsvRecord,
    n_windows: int,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> list[EegWindowMetadata]:
    return [
        EegWindowMetadata(
            subject=record.subject,
            game=record.game,
            label=record.label,
            source_file=record.source_file,
            start_sample=window_idx * window_samples,
            end_sample=(window_idx + 1) * window_samples,
        )
        for window_idx in range(n_windows)
    ]


def window_gameemo_record(
    record: GameemoCsvRecord,
    data: "np.ndarray",
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> tuple["np.ndarray", list[EegWindowMetadata]]:
    windows = window_eeg_array(data, window_samples=window_samples)
    metadata = build_window_metadata(record, windows.shape[0], window_samples=window_samples)
    return windows, metadata
