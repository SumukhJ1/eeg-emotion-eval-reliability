"""Train/test split helpers for EEG window metadata."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from src.windowing import EegWindowMetadata


@dataclass(frozen=True)
class MetadataSplit:
    train_indices: list[int]
    test_indices: list[int]
    split_name: str
    test_subject: str | None = None
    stratified: bool = False


def _validate_nonempty_metadata(metadata: Sequence[EegWindowMetadata]) -> None:
    if not metadata:
        raise ValueError("metadata must contain at least one window")


def _test_count(n_items: int, test_size: float) -> int:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    return max(1, min(n_items - 1, round(n_items * test_size)))


def _random_split_indices(n_items: int, test_size: float, rng: random.Random) -> tuple[list[int], list[int]]:
    indices = list(range(n_items))
    rng.shuffle(indices)
    n_test = _test_count(n_items, test_size)
    test_indices = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])
    return train_indices, test_indices


def _stratified_split_indices(
    metadata: Sequence[EegWindowMetadata],
    test_size: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(metadata):
        by_label[item.label].append(idx)

    if any(len(indices) < 2 for indices in by_label.values()):
        raise ValueError("stratification requires at least two windows per label")

    test_indices: list[int] = []
    train_indices: list[int] = []
    for indices in by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_test = _test_count(len(shuffled), test_size)
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    return sorted(train_indices), sorted(test_indices)


def make_subject_dependent_split(
    metadata: Sequence[EegWindowMetadata],
    test_size: float = 0.2,
    random_state: int = 0,
    stratify: bool = True,
) -> MetadataSplit:
    """Randomly split windows, optionally preserving label balance when feasible."""
    _validate_nonempty_metadata(metadata)
    if len(metadata) < 2:
        raise ValueError("at least two windows are required for a train/test split")

    rng = random.Random(random_state)
    if stratify:
        try:
            train_indices, test_indices = _stratified_split_indices(metadata, test_size, rng)
            return MetadataSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                split_name="subject_dependent",
                stratified=True,
            )
        except ValueError:
            pass

    train_indices, test_indices = _random_split_indices(len(metadata), test_size, rng)
    return MetadataSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        split_name="subject_dependent",
        stratified=False,
    )


def validate_loso_split(metadata: Sequence[EegWindowMetadata], split: MetadataSplit) -> None:
    if split.test_subject is None:
        raise ValueError("LOSO split must include test_subject")

    train_subjects = {metadata[idx].subject for idx in split.train_indices}
    test_subjects = {metadata[idx].subject for idx in split.test_indices}
    if test_subjects != {split.test_subject}:
        raise ValueError(f"expected only test subject {split.test_subject}, got {sorted(test_subjects)}")
    if split.test_subject in train_subjects:
        raise ValueError(f"LOSO leakage: subject {split.test_subject} appears in train and test")
    if not split.train_indices:
        raise ValueError(f"LOSO fold {split.test_subject} has no training windows")
    if not split.test_indices:
        raise ValueError(f"LOSO fold {split.test_subject} has no test windows")


def make_loso_splits(metadata: Sequence[EegWindowMetadata]) -> list[MetadataSplit]:
    """Create one fold per subject, holding out the entire subject for testing."""
    _validate_nonempty_metadata(metadata)
    subjects = sorted({item.subject for item in metadata})
    splits: list[MetadataSplit] = []

    for subject in subjects:
        train_indices = [idx for idx, item in enumerate(metadata) if item.subject != subject]
        test_indices = [idx for idx, item in enumerate(metadata) if item.subject == subject]
        split = MetadataSplit(
            train_indices=train_indices,
            test_indices=test_indices,
            split_name=f"loso_{subject}",
            test_subject=subject,
            stratified=False,
        )
        validate_loso_split(metadata, split)
        splits.append(split)

    return splits


def label_counts(metadata: Sequence[EegWindowMetadata], indices: Sequence[int]) -> Counter[str]:
    return Counter(metadata[idx].label for idx in indices)


def subject_counts(metadata: Sequence[EegWindowMetadata], indices: Sequence[int]) -> Counter[str]:
    return Counter(metadata[idx].subject for idx in indices)
