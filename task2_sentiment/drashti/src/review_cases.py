"""Choose the 20 review cases from stored predictions.

The groups are disjoint. Selection uses sort keys only.
"""

from __future__ import annotations

from typing import Any


LENGTH_SLICES = ("short", "medium", "long")


def _confidence(row: dict[str, Any]) -> float:
    if int(row["predicted_label"]) == 1:
        return float(row["positive_probability"])
    return float(row["negative_probability"])


def _take(rows: list[dict[str, Any]], count: int, key, reason: str, group: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=key)
    chosen = []
    for row in ordered[:count]:
        item = dict(row)
        item["review_group"] = group
        item["selection_reason"] = reason
        item["distance_from_threshold"] = abs(float(row["positive_probability"]) - 0.5)
        chosen.append(item)
    if len(chosen) != count:
        raise ValueError(f"{group} produced {len(chosen)} cases instead of {count}.")
    return chosen


def select_review_model(model_scores: list[tuple[str, float]]) -> str:
    """Highest official-test macro F1. Ties keep the earlier listed model."""
    if not model_scores:
        raise ValueError("Review model selection needs at least one score.")
    selected = model_scores[0]
    for name, score in model_scores[1:]:
        if score > selected[1]:
            selected = (name, score)
    return selected[0]


def select_review_cases(
    rows: list[dict[str, Any]],
    length_error_rates: dict[str, float],
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return 20 disjoint misclassified cases in the required groups."""
    blocked = excluded or set()
    rows = [row for row in rows if row["example_id"] not in blocked]
    false_positives = [row for row in rows if int(row["true_label"]) == 0 and int(row["predicted_label"]) == 1]
    false_negatives = [row for row in rows if int(row["true_label"]) == 1 and int(row["predicted_label"]) == 0]
    selected = _take(
        false_positives,
        5,
        lambda row: (-float(row["positive_probability"]), row["example_id"]),
        "Highest positive-class probability among reviews labeled negative and predicted positive.",
        "confident_false_positive",
    )
    selected.extend(
        _take(
            false_negatives,
            5,
            lambda row: (float(row["positive_probability"]), row["example_id"]),
            "Lowest positive-class probability among reviews labeled positive and predicted negative.",
            "confident_false_negative",
        )
    )
    used = {row["example_id"] for row in selected}
    errors = [row for row in rows if int(row["true_label"]) != int(row["predicted_label"]) and row["example_id"] not in used]
    selected.extend(
        _take(
            errors,
            5,
            lambda row: (abs(float(row["positive_probability"]) - 0.5), row["example_id"]),
            "Misclassified review whose positive-class probability is closest to the 0.5 threshold.",
            "near_threshold",
        )
    )
    used = {row["example_id"] for row in selected}
    remaining = [row for row in rows if int(row["true_label"]) != int(row["predicted_label"]) and row["example_id"] not in used]
    slice_cases = []
    for slice_name in LENGTH_SLICES:
        pool = [row for row in remaining if row["length_slice"] == slice_name]
        taken = _take(
            pool,
            1,
            lambda row: (-_confidence(row), row["example_id"]),
            f"Highest-confidence misclassification in the {slice_name} length slice.",
            "slice_specific",
        )
        slice_cases.extend(taken)
        remaining = [row for row in remaining if row["example_id"] != taken[0]["example_id"]]
    negation_pool = [row for row in remaining if int(row["contains_negation"]) == 1]
    taken = _take(
        negation_pool,
        1,
        lambda row: (-_confidence(row), row["example_id"]),
        "Highest-confidence misclassification whose processed text contains a negation token.",
        "slice_specific",
    )
    slice_cases.extend(taken)
    remaining = [row for row in remaining if row["example_id"] != taken[0]["example_id"]]
    worst_slice = max(LENGTH_SLICES, key=lambda name: (length_error_rates[name], name))
    worst_pool = [row for row in remaining if row["length_slice"] == worst_slice]
    if not worst_pool:
        worst_pool = remaining
        reason = "Highest-confidence remaining misclassification after the length and negation picks."
    else:
        reason = (
            f"Highest-confidence remaining misclassification in {worst_slice}, "
            "the length slice with the highest error rate."
        )
    slice_cases.extend(_take(worst_pool, 1, lambda row: (-_confidence(row), row["example_id"]), reason, "slice_specific"))
    selected.extend(slice_cases)
    if len(selected) != 20 or len({row["example_id"] for row in selected}) != 20:
        raise ValueError("The review selection did not produce 20 distinct examples.")
    return selected
