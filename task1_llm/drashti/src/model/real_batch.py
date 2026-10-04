"""Build one real TinyStories batch for the Phase 3 model check.

This uses the saved member split and vocabulary. It does not create synthetic stories.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lab1.paths import resolve_repo_path

from task1_llm.drashti.src.acquire import load_tinystories
from task1_llm.drashti.src.sequences import build_batch, pair_from_text
from task1_llm.drashti.src.vocab import vocabulary_from_payload


def load_real_training_batch(config: dict, repo: Path, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return input ids, target ids, and the vocabulary size from real training stories."""
    vocab_path = resolve_repo_path(config["vocab_metadata"], repo=repo)
    split_path = resolve_repo_path(config["split_metadata"], repo=repo)
    vocabulary = vocabulary_from_payload(json.loads(vocab_path.read_text(encoding="utf-8")))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dataset, _revision = load_tinystories(config, repo)
    wanted = config["batch_size"] if batch_size is None else batch_size
    pairs: list[tuple[list[int], list[int]]] = []
    text_field = config["text_field"]
    sequence_length = config["sequence_length"]
    for index in split["train_indices"]:
        text = dataset[int(index)][text_field]
        if not isinstance(text, str):
            raise TypeError("A selected TinyStories row did not contain string text.")
        pair = pair_from_text(text, vocabulary, sequence_length)
        if pair is None:
            continue
        pairs.append(pair)
        if len(pairs) == wanted:
            break
    if len(pairs) != wanted:
        raise RuntimeError("Not enough real training stories were long enough for a batch.")
    batch = build_batch(pairs)
    inputs = torch.tensor(batch["input_ids"], dtype=torch.long)
    targets = torch.tensor(batch["target_ids"], dtype=torch.long)
    return inputs, targets, vocabulary.size
