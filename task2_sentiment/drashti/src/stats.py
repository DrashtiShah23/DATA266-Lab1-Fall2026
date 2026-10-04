"""Length summaries. Percentiles use the nearest-rank method."""

from __future__ import annotations

import math


def nearest_rank(sorted_values: list[int], percentile: float) -> int:
    """Return the nearest-rank percentile of an ascending sequence."""
    if not sorted_values:
        raise ValueError("Cannot summarize an empty length list.")
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]
    rank = math.ceil(percentile / 100.0 * len(sorted_values)) - 1
    rank = min(max(rank, 0), len(sorted_values) - 1)
    return sorted_values[rank]


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    """Return minimum, maximum, mean, population standard deviation, and percentiles."""
    if not values:
        raise ValueError("Cannot summarize an empty length list.")
    ordered = sorted(values)
    count = len(ordered)
    total = sum(ordered)
    mean = total / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    summary: dict[str, float | int] = {
        "count": count,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": mean,
        "population_standard_deviation": math.sqrt(variance),
        "median": nearest_rank(ordered, 50),
    }
    for percentile in (25, 50, 75, 90, 95, 99):
        summary[f"percentile_{percentile}"] = nearest_rank(ordered, percentile)
    return summary
