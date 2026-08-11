"""Window-level EEG feature extraction."""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


STAT_FEATURES = ("mean", "std", "min", "max", "energy")


def make_stat_feature_names(channels: Sequence[str]) -> list[str]:
    return [f"{channel}_{feature}" for channel in channels for feature in STAT_FEATURES]


def extract_statistical_features(
    windows: "np.ndarray",
    channels: Sequence[str] | None = None,
) -> tuple["np.ndarray", list[str]]:
    """Extract simple per-channel statistics from EEG windows.

    Parameters
    ----------
    windows:
        EEG windows shaped n_windows x channels x window_samples.
    channels:
        Optional channel names used to build feature names. If omitted, names
        default to channel_0, channel_1, ...

    Returns
    -------
    features, feature_names:
        `features` is shaped n_windows x n_features. The feature set is a
        first-pass baseline: mean, standard deviation, min, max, and mean
        squared value per channel.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required to extract EEG window features.") from exc

    if windows.ndim != 3:
        raise ValueError(f"Expected windows shaped n_windows x channels x samples, got {windows.shape}")

    n_windows, n_channels, _ = windows.shape
    if channels is None:
        channel_names = [f"channel_{idx}" for idx in range(n_channels)]
    else:
        channel_names = list(channels)
        if len(channel_names) != n_channels:
            raise ValueError(f"Expected {n_channels} channel names, got {len(channel_names)}")

    stats = [
        windows.mean(axis=2),
        windows.std(axis=2),
        windows.min(axis=2),
        windows.max(axis=2),
        np.mean(np.square(windows), axis=2),
    ]
    features = np.stack(stats, axis=2).reshape(n_windows, n_channels * len(STAT_FEATURES))
    return features, make_stat_feature_names(channel_names)
