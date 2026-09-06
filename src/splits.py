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


@dataclass(frozen=True)
class BudgetedMetadataSplit:
    fit_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    split_name: str
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


def _label_target_counts(
    metadata: Sequence[EegWindowMetadata],
    candidate_indices: Sequence[int],
    target_count: int,
) -> dict[str, int]:
    if target_count < 0:
        raise ValueError(f"target_count must be non-negative, got {target_count}")
    if target_count > len(candidate_indices):
        raise ValueError(f"target_count={target_count} exceeds candidate count={len(candidate_indices)}")

    by_label = label_counts(metadata, candidate_indices)
    total = sum(by_label.values())
    if target_count == 0:
        return {label: 0 for label in by_label}
    if total == 0:
        raise ValueError("cannot allocate target counts from an empty candidate set")

    raw_targets = {
        label: (count * target_count) / total
        for label, count in by_label.items()
    }
    targets = {label: int(value) for label, value in raw_targets.items()}
    remaining = target_count - sum(targets.values())
    labels_by_remainder = sorted(
        raw_targets,
        key=lambda label: (raw_targets[label] - targets[label], by_label[label], label),
        reverse=True,
    )
    for label in labels_by_remainder[:remaining]:
        targets[label] += 1

    for label, target in targets.items():
        if target > by_label[label]:
            raise ValueError(f"cannot sample {target} items for label {label}; only {by_label[label]} available")
    return targets


def _sample_stratified_exact(
    metadata: Sequence[EegWindowMetadata],
    candidate_indices: Sequence[int],
    target_count: int,
    rng: random.Random,
) -> list[int]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx in candidate_indices:
        by_label[metadata[idx].label].append(idx)

    targets = _label_target_counts(metadata, candidate_indices, target_count)
    selected: list[int] = []
    for label, indices in by_label.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[: targets[label]])
    return sorted(selected)


def _sample_random_exact(candidate_indices: Sequence[int], target_count: int, rng: random.Random) -> list[int]:
    if target_count > len(candidate_indices):
        raise ValueError(f"target_count={target_count} exceeds candidate count={len(candidate_indices)}")
    shuffled = list(candidate_indices)
    rng.shuffle(shuffled)
    return sorted(shuffled[:target_count])


def _sample_exact(
    metadata: Sequence[EegWindowMetadata],
    candidate_indices: Sequence[int],
    target_count: int,
    rng: random.Random,
    stratify: bool,
) -> tuple[list[int], bool]:
    if stratify:
        try:
            return _sample_stratified_exact(metadata, candidate_indices, target_count, rng), True
        except ValueError:
            pass
    return _sample_random_exact(candidate_indices, target_count, rng), False


def make_subject_dependent_matched_budget_split(
    metadata: Sequence[EegWindowMetadata],
    *,
    fit_windows: int,
    val_windows: int,
    test_windows: int,
    random_state: int = 0,
    stratify: bool = True,
) -> BudgetedMetadataSplit:
    """Create a random window split with explicit fit/validation/test budgets.

    This is useful for comparing subject-dependent experiments against LOSO
    folds without giving one protocol a larger training budget.
    """
    _validate_nonempty_metadata(metadata)
    if min(fit_windows, val_windows, test_windows) <= 0:
        raise ValueError("fit_windows, val_windows, and test_windows must all be positive")
    total_requested = fit_windows + val_windows + test_windows
    if total_requested > len(metadata):
        raise ValueError(
            f"requested {total_requested} windows, but metadata contains only {len(metadata)} windows"
        )

    rng = random.Random(random_state)
    all_indices = list(range(len(metadata)))
    test_indices, test_stratified = _sample_exact(metadata, all_indices, test_windows, rng, stratify)
    remaining_after_test = [idx for idx in all_indices if idx not in set(test_indices)]
    val_indices, val_stratified = _sample_exact(
        metadata,
        remaining_after_test,
        val_windows,
        rng,
        stratify,
    )
    reserved = set(test_indices) | set(val_indices)
    fit_candidates = [idx for idx in all_indices if idx not in reserved]
    fit_indices, fit_stratified = _sample_exact(metadata, fit_candidates, fit_windows, rng, stratify)

    return BudgetedMetadataSplit(
        fit_indices=fit_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_name="subject_dependent_matched_budget",
        stratified=test_stratified and val_stratified and fit_stratified,
    )


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
