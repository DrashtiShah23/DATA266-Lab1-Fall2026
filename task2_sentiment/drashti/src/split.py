"""Development and validation indices carved only from the official train split."""

from __future__ import annotations

import random


def development_validation_indices(train_count: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Return sorted development and validation indices.

    ``random.Random(seed)`` shuffles ``range(train_count)``. Validation is the
    first ``floor(fraction * train_count)`` shuffled indices. The official test
    split is not an input. Both returned lists are sorted so later scans follow
    official row order. Sorting does not change membership.
    """
    if isinstance(train_count, bool) or not isinstance(train_count, int) or train_count < 2:
        raise ValueError("train_count must be an integer of at least 2.")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < float(fraction) < 1:
        raise ValueError("fraction must be strictly between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    validation_count = int(train_count * float(fraction))
    if validation_count < 1 or validation_count >= train_count:
        raise ValueError("The validation split would be empty or would consume every training row.")
    order = list(range(train_count))
    random.Random(seed).shuffle(order)
    validation = sorted(order[:validation_count])
    development = sorted(order[validation_count:])
    return development, validation


def split_is_partition(train_count: int, development: list[int], validation: list[int]) -> bool:
    """Return whether the two lists are disjoint and cover every train index once."""
    if len(development) + len(validation) != train_count:
        return False
    if set(development) & set(validation):
        return False
    return set(development) | set(validation) == set(range(train_count))
