"""Deterministic member split for Drashti's Task 1 data.

The split selects stories. It does not create synthetic text.
"""

from __future__ import annotations

import random

SELECTION_PROCEDURE = (
    "Use the official train split of roneneldan/TinyStories as the source pool. "
    "Build range(source_record_count) and shuffle it in place with "
    "random.Random(split_seed).shuffle under CPython 3.11. "
    "That shuffle is Fisher-Yates driven by CPython's random generator. "
    "Training indices are the first train_size items of the shuffled list. "
    "Validation indices are the next validation_size items. "
    "The slices are disjoint because they occupy different positions in one permutation. "
    "Index order inside each slice is the shuffled order."
)


def member_split_indices(
    source_size: int,
    train_size: int,
    validation_size: int,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Return disjoint training and validation source indices."""
    _require_count("source_size", source_size)
    _require_count("train_size", train_size)
    _require_count("validation_size", validation_size)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("split_seed must be an integer.")
    needed = train_size + validation_size
    if source_size < needed:
        raise ValueError("Source split does not contain enough records for the member split.")
    indices = list(range(source_size))
    random.Random(split_seed).shuffle(indices)
    training = indices[:train_size]
    validation = indices[train_size:needed]
    return training, validation


def disjointness_report(training: list[int], validation: list[int]) -> dict[str, int | bool]:
    """Count overlap. This inspects the index lists; it does not invent a result."""
    training_set = set(training)
    validation_set = set(validation)
    overlap = training_set & validation_set
    return {
        "train_count": len(training),
        "validation_count": len(validation),
        "train_unique": len(training_set) == len(training),
        "validation_unique": len(validation_set) == len(validation),
        "disjoint": len(overlap) == 0,
        "overlap_count": len(overlap),
    }


def _require_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
