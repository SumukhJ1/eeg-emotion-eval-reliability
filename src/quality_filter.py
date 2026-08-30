"""Conservative EEG window quality filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


DEFAULT_VARIANCE_EPSILON = 1e-12


@dataclass(frozen=True)
class QualityFilterReport:
    n_windows: int
    kept_windows: int
    removed_windows: int
    missing_value_windows: int
    nonfinite_value_windows: int
    near_zero_variance_windows: int
    flat_channel_windows: int
    flat_channel_instances: int
    variance_epsilon: float = DEFAULT_VARIANCE_EPSILON


def conservative_quality_mask(
    windows: "np.ndarray",
    variance_epsilon: float = DEFAULT_VARIANCE_EPSILON,
) -> tuple["np.ndarray", QualityFilterReport]:
    """Return a keep mask using the pre-registered conservative filtering rules.

    Removed windows have at least one missing/nonfinite value, near-zero whole-window
    variance, or at least one flat channel. Extreme-amplitude and extreme-variance
    windows are deliberately not removed by this mask.
    """
    import numpy as np

    if windows.ndim != 3:
        raise ValueError(f"Expected windows shaped n_windows x channels x samples, got {windows.shape}")
    if variance_epsilon < 0:
        raise ValueError(f"variance_epsilon must be non-negative, got {variance_epsilon}")

    missing_windows = np.isnan(windows).any(axis=(1, 2))
    nonfinite_windows = (~np.isfinite(windows)).any(axis=(1, 2))
    channel_variance = np.nanvar(windows, axis=2)
    window_variance = np.nanvar(windows, axis=(1, 2))
    near_zero_windows = window_variance <= variance_epsilon
    flat_channels = channel_variance <= variance_epsilon
    flat_channel_windows = flat_channels.any(axis=1)

    remove_mask = missing_windows | nonfinite_windows | near_zero_windows | flat_channel_windows
    keep_mask = ~remove_mask
    report = QualityFilterReport(
        n_windows=int(windows.shape[0]),
        kept_windows=int(keep_mask.sum()),
        removed_windows=int(remove_mask.sum()),
        missing_value_windows=int(missing_windows.sum()),
        nonfinite_value_windows=int(nonfinite_windows.sum()),
        near_zero_variance_windows=int(near_zero_windows.sum()),
        flat_channel_windows=int(flat_channel_windows.sum()),
        flat_channel_instances=int(flat_channels.sum()),
        variance_epsilon=variance_epsilon,
    )
    return keep_mask, report
