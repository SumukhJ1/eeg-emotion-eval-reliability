"""Window-level EEG feature extraction."""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


STAT_FEATURES = ("mean", "std", "min", "max", "energy")
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BANDPOWER_MODES = ("absolute", "log", "relative", "log_relative")


def make_stat_feature_names(channels: Sequence[str]) -> list[str]:
    return [f"{channel}_{feature}" for channel in channels for feature in STAT_FEATURES]


def _channel_names(channels: Sequence[str] | None, n_channels: int) -> list[str]:
    if channels is None:
        return [f"channel_{idx}" for idx in range(n_channels)]

    channel_names = list(channels)
    if len(channel_names) != n_channels:
        raise ValueError(f"Expected {n_channels} channel names, got {len(channel_names)}")
    return channel_names


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
    channel_names = _channel_names(channels, n_channels)

    stats = [
        windows.mean(axis=2),
        windows.std(axis=2),
        windows.min(axis=2),
        windows.max(axis=2),
        np.mean(np.square(windows), axis=2),
    ]
    features = np.stack(stats, axis=2).reshape(n_windows, n_channels * len(STAT_FEATURES))
    return features, make_stat_feature_names(channel_names)


def make_bandpower_feature_names(
    channels: Sequence[str],
    bands: dict[str, tuple[float, float]] = EEG_BANDS,
    mode: str = "absolute",
) -> list[str]:
    if mode not in BANDPOWER_MODES:
        raise ValueError(f"mode must be one of {BANDPOWER_MODES}, got {mode}")
    return [f"{channel}_{band}_{mode}_power" for channel in channels for band in bands]


def extract_bandpower_features(
    windows: "np.ndarray",
    channels: Sequence[str] | None = None,
    sampling_rate: int = 128,
    bands: dict[str, tuple[float, float]] = EEG_BANDS,
    mode: str = "absolute",
    epsilon: float = 1e-12,
) -> tuple["np.ndarray", list[str]]:
    """Extract simple FFT bandpower features from EEG windows.

    This is an EEG-specific baseline feature set, not a final tuned signal
    processing pipeline. Inputs are windows shaped n_windows x channels x
    window_samples. Outputs are shaped n_windows x (channels * bands).

    Supported modes:

    - absolute: mean FFT power in each band.
    - log: log absolute bandpower.
    - relative: bandpower divided by total modeled bandpower per channel/window.
    - log_relative: log relative bandpower.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required to extract EEG bandpower features.") from exc

    if windows.ndim != 3:
        raise ValueError(f"Expected windows shaped n_windows x channels x samples, got {windows.shape}")
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    if not bands:
        raise ValueError("bands must contain at least one frequency range")
    if mode not in BANDPOWER_MODES:
        raise ValueError(f"mode must be one of {BANDPOWER_MODES}, got {mode}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    n_windows, n_channels, n_samples = windows.shape
    channel_names = _channel_names(channels, n_channels)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sampling_rate)
    spectrum = np.fft.rfft(windows, axis=2)
    power = (np.abs(spectrum) ** 2) / n_samples

    band_blocks = []
    for band_name, (low_hz, high_hz) in bands.items():
        if low_hz < 0 or high_hz <= low_hz:
            raise ValueError(f"Invalid band {band_name}: {(low_hz, high_hz)}")
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not mask.any():
            raise ValueError(f"Band {band_name} has no FFT bins for {n_samples} samples at {sampling_rate} Hz")
        band_blocks.append(power[:, :, mask].mean(axis=2))

    bandpower = np.stack(band_blocks, axis=2)
    if mode == "absolute":
        transformed = bandpower
    elif mode == "log":
        transformed = np.log(bandpower + epsilon)
    elif mode == "relative":
        total_power = bandpower.sum(axis=2, keepdims=True)
        transformed = bandpower / (total_power + epsilon)
    else:
        total_power = bandpower.sum(axis=2, keepdims=True)
        relative_power = bandpower / (total_power + epsilon)
        transformed = np.log(relative_power + epsilon)

    features = transformed.reshape(n_windows, n_channels * len(bands))
    return features, make_bandpower_feature_names(channel_names, bands=bands, mode=mode)


def extract_log_relative_bandpower_features(
    windows: "np.ndarray",
    channels: Sequence[str] | None = None,
    sampling_rate: int = 128,
    bands: dict[str, tuple[float, float]] = EEG_BANDS,
) -> tuple["np.ndarray", list[str]]:
    """Extract log-relative FFT bandpower features from EEG windows."""
    return extract_bandpower_features(
        windows,
        channels=channels,
        sampling_rate=sampling_rate,
        bands=bands,
        mode="log_relative",
    )
