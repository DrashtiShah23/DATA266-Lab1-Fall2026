"""Length and negation slices for later robustness analysis.

No model score is computed here. Thresholds come from the development-training
token counts so the official test split does not choose the cut points.
"""

from __future__ import annotations


def length_slice(token_count: int, low_cutoff: int, high_cutoff: int) -> str:
    """Assign one length slice. Empty processed reviews are their own slice."""
    if token_count < 0:
        raise ValueError("token_count cannot be negative.")
    if token_count == 0:
        return "empty_after_preprocessing"
    if token_count <= low_cutoff:
        return "short"
    if token_count <= high_cutoff:
        return "medium"
    return "long"


def assign_slices(token_count: int, contains_negation: bool, low_cutoff: int, high_cutoff: int) -> list[str]:
    """Return the length slice and, when present, the overlapping negation slice."""
    assigned = [length_slice(token_count, low_cutoff, high_cutoff)]
    if contains_negation:
        assigned.append("contains_negation")
    return assigned
