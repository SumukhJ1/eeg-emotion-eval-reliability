"""Train-split-only normalization helpers for neural EEG baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ChannelStandardizationStats:
    """Per-channel statistics estimated from training windows only."""

    mean: "np.ndarray"
    std: "np.ndarray"
    strategy: str = "train_channel_standardization"


def fit_channel_standardization(
    windows: "np.ndarray",
    train_indices: list[int] | "np.ndarray",
) -> ChannelStandardizationStats:
    """Fit per-channel mean/std using only the training split.

    Windows are expected to be shaped n_windows x n_channels x n_samples.
    Statistics are pooled over training windows and time samples, separately
    for each EEG channel, then reused for validation/test windows.
    """
    train_block = windows[train_indices]
    mean = train_block.mean(axis=(0, 2), keepdims=True)
    std = train_block.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return ChannelStandardizationStats(mean=mean, std=std)


def apply_channel_standardization(
    windows: "np.ndarray",
    stats: ChannelStandardizationStats,
) -> "np.ndarray":
    """Apply previously fitted train-only channel standardization."""
    return (windows - stats.mean) / stats.std


def channel_standardize_splits(
    windows: "np.ndarray",
    train_indices: list[int] | "np.ndarray",
    val_indices: list[int] | "np.ndarray",
    test_indices: list[int] | "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", ChannelStandardizationStats]:
    """Normalize train/validation/test windows without using held-out data."""
    stats = fit_channel_standardization(windows, train_indices)
    return (
        apply_channel_standardization(windows[train_indices], stats),
        apply_channel_standardization(windows[val_indices], stats),
        apply_channel_standardization(windows[test_indices], stats),
        stats,
    )
