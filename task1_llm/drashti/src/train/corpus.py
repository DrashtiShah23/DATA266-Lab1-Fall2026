"""Real TinyStories windows for training.

The member split stays 100,000 training stories and 10,000 validation stories.
A story shorter than ``sequence_length + 1`` characters stays in that split and
is skipped. No characters are added to make it long enough. Each eligible story
contributes one window from its start, using the Phase 2 shift:
input is characters ``[0, sequence_length)`` and target is ``[1, sequence_length + 1)``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from lab1.paths import resolve_repo_path

from task1_llm.drashti.src.acquire import load_tinystories
from task1_llm.drashti.src.vocab import vocabulary_from_payload

CACHE_RELATIVE = "task1_llm/drashti/data_processed/sequence_windows.pt"
SHORT_STORY_POLICY = (
    "Keep all 100000 training stories and all 10000 validation stories in the member split. "
    "Skip a story when it has fewer than sequence_length + 1 characters. "
    "Do not pad, repeat, or invent characters. "
    "Each eligible story contributes the first window only."
)


def describe_split(selected_count: int, character_lengths: list[int], sequence_length: int) -> dict[str, int]:
    """Report the selected split size separately from the eligible window count."""
    if selected_count != len(character_lengths):
        raise ValueError("character_lengths must contain one entry per selected story.")
    minimum = sequence_length + 1
    eligible = sum(1 for length in character_lengths if length >= minimum)
    return {
        "selected_stories": selected_count,
        "eligible_windows": eligible,
        "skipped_short_stories": selected_count - eligible,
    }


def load_corpus(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Load cached windows or build them from the real split."""
    cache_path = resolve_repo_path(CACHE_RELATIVE, repo=repo)
    if cache_path.is_file():
        cached = torch.load(cache_path, weights_only=True)
        if _cache_matches(cached, config):
            return cached
    built = build_corpus(config, repo)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(built, cache_path)
    return built


def build_corpus(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Encode the saved member split into fixed windows."""
    vocabulary = vocabulary_from_payload(
        json.loads(resolve_repo_path(config["vocab_metadata"], repo=repo).read_text(encoding="utf-8"))
    )
    split = json.loads(resolve_repo_path(config["split_metadata"], repo=repo).read_text(encoding="utf-8"))
    if len(split["train_indices"]) != config["train_size"] or len(split["validation_indices"]) != config["validation_size"]:
        raise RuntimeError("Saved split counts do not match the config.")
    if split["dataset_revision"] != config["dataset_revision"] or split["split_seed"] != config["split_seed"]:
        raise RuntimeError("Saved split does not match the training config revision or seed.")
    dataset, revision = load_tinystories(config, repo)
    if revision != config["dataset_revision"]:
        raise RuntimeError("Loaded TinyStories revision does not match the config.")
    sequence_length = int(config["sequence_length"])
    train_windows, train_skipped = _windows_for_indices(
        dataset, split["train_indices"], config["text_field"], vocabulary, sequence_length
    )
    validation_windows, validation_skipped = _windows_for_indices(
        dataset, split["validation_indices"], config["text_field"], vocabulary, sequence_length
    )
    return {
        "dataset_revision": revision,
        "split_seed": int(config["split_seed"]),
        "sequence_length": sequence_length,
        "vocab_size": int(vocabulary.size),
        "train_selected": int(config["train_size"]),
        "validation_selected": int(config["validation_size"]),
        "train_skipped": train_skipped,
        "validation_skipped": validation_skipped,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "policy": SHORT_STORY_POLICY,
    }


def prompts_from_validation(
    dataset: Any,
    validation_indices: list[int],
    text_field: str,
    prompt_characters: int,
    sample_count: int,
) -> list[str]:
    """Take prompt prefixes from validation stories in split order."""
    prompts: list[str] = []
    for index in validation_indices:
        text = dataset[int(index)][text_field]
        if not isinstance(text, str):
            raise TypeError("A validation row did not contain string text.")
        if len(text) < prompt_characters:
            continue
        prompts.append(text[:prompt_characters])
        if len(prompts) == sample_count:
            return prompts
    raise RuntimeError("Not enough validation stories were long enough to supply prompts.")


def _windows_for_indices(dataset: Any, indices: list[int], text_field: str, vocabulary: Any, sequence_length: int) -> tuple[torch.Tensor, int]:
    minimum = sequence_length + 1
    rows: list[list[int]] = []
    skipped = 0
    for position, index in enumerate(indices):
        text = dataset[int(index)][text_field]
        if not isinstance(text, str):
            raise TypeError("A selected TinyStories row did not contain string text.")
        if len(text) < minimum:
            skipped += 1
            continue
        token_ids = vocabulary.encode(text)
        if len(token_ids) < minimum:
            skipped += 1
            continue
        rows.append(token_ids[:minimum])
        if position and position % 20000 == 0:
            print(f"encoded {position} selected stories", flush=True)
    if not rows:
        raise RuntimeError("No eligible windows were built.")
    return torch.tensor(rows, dtype=torch.int32), skipped


def iter_windows(
    windows: torch.Tensor,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield input and target rows. The last batch may be smaller than ``batch_size``."""
    count = int(windows.shape[0])
    order = list(range(count))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    for start in range(0, count, batch_size):
        index = torch.tensor(order[start : start + batch_size], dtype=torch.long)
        chunk = windows.index_select(0, index).to(dtype=torch.long)
        yield chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


def steps_per_pass(num_windows: int, batch_size: int) -> int:
    """Count batches, including a smaller final batch when the count does not divide evenly."""
    if num_windows < 1 or batch_size < 1:
        raise ValueError("Window count and batch size must be positive.")
    return (num_windows + batch_size - 1) // batch_size


def _cache_matches(cached: dict[str, Any], config: dict[str, Any]) -> bool:
    required = (
        "dataset_revision",
        "split_seed",
        "sequence_length",
        "vocab_size",
        "train_selected",
        "validation_selected",
        "train_windows",
        "validation_windows",
    )
    if any(key not in cached for key in required):
        return False
    return (
        cached["dataset_revision"] == config["dataset_revision"]
        and int(cached["split_seed"]) == int(config["split_seed"])
        and int(cached["sequence_length"]) == int(config["sequence_length"])
        and int(cached["train_selected"]) == int(config["train_size"])
        and int(cached["validation_selected"]) == int(config["validation_size"])
    )
