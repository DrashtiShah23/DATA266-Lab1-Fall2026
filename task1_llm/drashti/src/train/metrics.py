"""Task 1 metric formulas.

Cross-entropy values are in nats, matching ``torch.nn.functional.cross_entropy``.
"""

from __future__ import annotations

import math
from collections import Counter


def perplexity(validation_cross_entropy: float) -> float:
    """``exp(validation cross entropy)``."""
    return math.exp(validation_cross_entropy)


def bits_per_character(validation_cross_entropy: float) -> float:
    """``validation cross entropy / ln(2)``."""
    return validation_cross_entropy / math.log(2.0)


def generalization_gap(validation_cross_entropy: float, training_cross_entropy: float) -> float:
    """Validation cross entropy minus training cross entropy."""
    return validation_cross_entropy - training_cross_entropy


def top1_accuracy(correct_predictions: int, evaluated_targets: int) -> float:
    """Correct argmax predictions divided by evaluated targets."""
    if evaluated_targets < 1:
        raise ValueError("evaluated_targets must be positive.")
    return correct_predictions / evaluated_targets


def ngrams(text: str, order: int) -> list[str]:
    """Contiguous character n-grams. Order is the character count."""
    if order < 1:
        raise ValueError("n-gram order must be positive.")
    if len(text) < order:
        return []
    return [text[index : index + order] for index in range(len(text) - order + 1)]


def distinct_n(texts: list[str], order: int) -> float:
    """Unique character n-grams divided by total character n-grams.

    N-grams are pooled across the supplied strings. Each string is one
    generated continuation, excluding its prompt.
    """
    total: list[str] = []
    for text in texts:
        total.extend(ngrams(text, order))
    if not total:
        raise ValueError("No n-grams were available.")
    return len(set(total)) / len(total)


REPEATED_4GRAM_DEFINITION = (
    "For one generated continuation, a 4-gram position is repeated when the same "
    "four characters occurred at a strictly earlier position in that same continuation. "
    "The sample rate is repeated positions divided by the number of 4-gram positions. "
    "The reported rate is the unweighted mean of the sample rates. Prompts are excluded."
)


def repeated_4gram_rate(text: str) -> float:
    """Fraction of 4-gram positions that repeat an earlier 4-gram in the same text.

    A position counts as repeated when its 4 characters already occurred at a
    strictly earlier position in that same string. The rate is
    ``repeated_positions / total_4gram_positions``.
    """
    grams = ngrams(text, 4)
    if not grams:
        raise ValueError("Text is shorter than 4 characters.")
    seen: set[str] = set()
    repeated = 0
    for gram in grams:
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / len(grams)


def mean_repeated_4gram_rate(texts: list[str]) -> float:
    """Mean of :func:`repeated_4gram_rate` over texts that contain a 4-gram."""
    rates = [repeated_4gram_rate(text) for text in texts if len(text) >= 4]
    if not rates:
        raise ValueError("No text contained a 4-gram.")
    return sum(rates) / len(rates)


def loss_spike_count(step_losses: list[float], margin_nats: float) -> int:
    """Count steps whose loss is at least ``margin_nats`` above the epoch median.

    The median is the middle value of the sorted step losses from that same
    epoch. A spike is a step with ``loss >= median + margin_nats``.
    """
    if not step_losses:
        return 0
    if margin_nats < 0:
        raise ValueError("margin_nats must be >= 0.")
    ordered = sorted(step_losses)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    threshold = median + margin_nats
    return sum(1 for loss in step_losses if loss >= threshold)


def finite_number(value: float) -> bool:
    """Return whether ``value`` is a finite float."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ngram_counts(text: str, order: int) -> Counter[str]:
    return Counter(ngrams(text, order))
