"""Model-ready access for Phase 6.

Token ids are integers. This module does not create or load embedding vectors.
The original review is read back from the official dataset with the example id.
"""

from __future__ import annotations

from typing import Any


def example_id(split: str, index: int) -> str:
    """Return a stable id for an official Yelp row."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test.")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer.")
    return f"{split}-{index}"


def parse_example_id(example_id_text: str) -> tuple[str, int]:
    """Invert ``example_id``."""
    split, separator, index_text = example_id_text.partition("-")
    if separator != "-" or split not in ("train", "test") or not index_text.isdigit():
        raise ValueError("example id must look like train-0 or test-0.")
    return split, int(index_text)


def original_text(dataset: Any, example_id_text: str, text_field: str) -> str:
    """Load the official review text for a stable example id."""
    split, index = parse_example_id(example_id_text)
    text = dataset[split][index][text_field]
    if not isinstance(text, str):
        raise ValueError("The official row does not contain string text.")
    return text


def make_batch(split_cache: dict[str, Any], batch_size: int, id_prefix: str) -> dict[str, Any]:
    """Return the first ``batch_size`` model-ready rows of one cached split."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    count = int(split_cache["input_ids"].shape[0])
    if count < batch_size:
        raise ValueError("The cached split does not contain a full batch.")
    ids = split_cache["input_ids"][:batch_size]
    labels = split_cache["labels"][:batch_size]
    lengths = split_cache["content_length"][:batch_size]
    example_ids = [example_id(id_prefix, int(split_cache["source_index"][row])) for row in range(batch_size)]
    return {
        "input_ids": ids,
        "labels": labels,
        "content_length": lengths,
        "example_ids": example_ids,
    }
