"""Turn processed tokens into padded integer ids.

No embedding matrix is created. Phase 6 will initialize embeddings at random.
"""

from __future__ import annotations

from task2_sentiment.drashti.src.vocabulary import PAD_ID, Vocabulary


def numericalize(tokens: list[str], vocabulary: Vocabulary, maximum_length: int) -> dict[str, object]:
    """Truncate on the left-to-right prefix and pad on the right."""
    if isinstance(maximum_length, bool) or not isinstance(maximum_length, int) or maximum_length < 1:
        raise ValueError("maximum_length must be a positive integer.")
    ids = vocabulary.encode(tokens)
    token_count = len(ids)
    truncated = token_count > maximum_length
    if truncated:
        ids = ids[:maximum_length]
    content_length = len(ids)
    padded = ids + [PAD_ID] * (maximum_length - content_length)
    return {
        "input_ids": padded,
        "content_length": content_length,
        "token_count": token_count,
        "truncated": truncated,
        "padded": content_length < maximum_length,
    }


def convert_label(label: object, allowed: tuple[int, ...] = (0, 1)) -> int:
    """Keep an official integer label. Booleans and other values are rejected."""
    if isinstance(label, bool) or not isinstance(label, int) or label not in allowed:
        raise ValueError("Label is not one of the official Yelp Polarity class indices.")
    return label
