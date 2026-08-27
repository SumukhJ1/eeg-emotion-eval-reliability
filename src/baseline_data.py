"""Shared dataset builders for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.features import extract_statistical_features
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
