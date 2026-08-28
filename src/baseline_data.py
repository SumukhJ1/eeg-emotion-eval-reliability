"""Shared dataset builders for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import LABEL_MAP, SAMPLING_RATE
from src.features import EEG_BANDS, extract_bandpower_features, extract_statistical_features
from src.gameemo_loader import GAMEEMO_ROOT, discover_records, load_record
from src.windowing import DEFAULT_WINDOW_SAMPLES, EegWindowMetadata, window_gameemo_record

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class BaselineFeatureDataset:
    features: "np.ndarray"
    metadata: list[EegWindowMetadata]
    feature_names: list[str]
    n_records: int


@dataclass(frozen=True)
class BaselineWindowDataset:
    windows: "np.ndarray"
    labels: "np.ndarray"
    metadata: list[EegWindowMetadata]
    n_records: int


def build_window_dataset(
    root: Path = GAMEEMO_ROOT,
    limit_records: int | None = None,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> BaselineWindowDataset:
    """Load GAMEEMO CSVs into raw windows shaped windows x channels x samples."""
    import numpy as np

    records = discover_records(root, limit=limit_records)
    if not records:
        raise ValueError(f"No GAMEEMO preprocessed CSV records found under {root}")

    window_blocks = []
    metadata: list[EegWindowMetadata] = []

    for record in records:
        loaded_record, data = load_record(record)
        windows, window_metadata = window_gameemo_record(
            loaded_record,
            data,
            window_samples=window_samples,
        )
        window_blocks.append(windows)
        metadata.extend(window_metadata)

    labels = np.asarray([LABEL_MAP[item.label] for item in metadata], dtype=np.int64)
    return BaselineWindowDataset(
        windows=np.vstack(window_blocks),
        labels=labels,
        metadata=metadata,
        n_records=len(records),
    )


def build_stat_feature_dataset(
    root: Path = GAMEEMO_ROOT,
    limit_records: int | None = None,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
) -> BaselineFeatureDataset:
    """Load GAMEEMO CSVs and build statistical window features for baselines."""
    import numpy as np

    records = discover_records(root, limit=limit_records)
    if not records:
        raise ValueError(f"No GAMEEMO preprocessed CSV records found under {root}")

    feature_blocks = []
    metadata: list[EegWindowMetadata] = []
    feature_names: list[str] | None = None

    for record in records:
        loaded_record, data = load_record(record)
        windows, window_metadata = window_gameemo_record(
            loaded_record,
            data,
            window_samples=window_samples,
        )
        features, names = extract_statistical_features(windows, channels=loaded_record.channels)

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Feature names changed for {loaded_record.source_file}")

        feature_blocks.append(features)
        metadata.extend(window_metadata)

    return BaselineFeatureDataset(
        features=np.vstack(feature_blocks),
        metadata=metadata,
        feature_names=feature_names or [],
        n_records=len(records),
    )


def build_bandpower_feature_dataset(
    root: Path = GAMEEMO_ROOT,
    limit_records: int | None = None,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    sampling_rate: int = SAMPLING_RATE,
    mode: str = "absolute",
) -> BaselineFeatureDataset:
    """Load GAMEEMO CSVs and build FFT bandpower window features for baselines."""
    import numpy as np

    records = discover_records(root, limit=limit_records)
    if not records:
        raise ValueError(f"No GAMEEMO preprocessed CSV records found under {root}")

    feature_blocks = []
    metadata: list[EegWindowMetadata] = []
    feature_names: list[str] | None = None

    for record in records:
        loaded_record, data = load_record(record)
        windows, window_metadata = window_gameemo_record(
            loaded_record,
            data,
            window_samples=window_samples,
        )
        features, names = extract_bandpower_features(
            windows,
            channels=loaded_record.channels,
            sampling_rate=sampling_rate,
            bands=EEG_BANDS,
            mode=mode,
        )

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Feature names changed for {loaded_record.source_file}")

        feature_blocks.append(features)
        metadata.extend(window_metadata)

    return BaselineFeatureDataset(
        features=np.vstack(feature_blocks),
        metadata=metadata,
        feature_names=feature_names or [],
        n_records=len(records),
    )


def build_log_relative_bandpower_feature_dataset(
    root: Path = GAMEEMO_ROOT,
    limit_records: int | None = None,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    sampling_rate: int = SAMPLING_RATE,
) -> BaselineFeatureDataset:
    """Load GAMEEMO CSVs and build log-relative FFT bandpower window features."""
    return build_bandpower_feature_dataset(
        root=root,
        limit_records=limit_records,
        window_samples=window_samples,
        sampling_rate=sampling_rate,
        mode="log_relative",
    )
