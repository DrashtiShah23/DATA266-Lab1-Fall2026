"""Fixed-length autoregressive character sequences."""

from __future__ import annotations

from task1_llm.drashti.src.vocab import CharacterVocabulary


def autoregressive_window(
    token_ids: list[int],
    sequence_length: int,
) -> tuple[list[int], list[int]] | None:
    """Return input and next-character target from the start of ``token_ids``.

    input is token_ids[0:sequence_length].
    target is token_ids[1:sequence_length + 1].
    Stories shorter than sequence_length + 1 characters return None.
    """
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if len(token_ids) < sequence_length + 1:
        return None
    return token_ids[:sequence_length], token_ids[1 : sequence_length + 1]


def shift_matches(token_ids: list[int], inputs: list[int], targets: list[int]) -> bool:
    """Return whether target[t] is the token immediately after input[t]."""
    if len(inputs) != len(targets) or not inputs:
        return False
    sequence_length = len(inputs)
    if token_ids[:sequence_length] != inputs:
        return False
    if token_ids[1 : sequence_length + 1] != targets:
        return False
    return all(targets[position] == token_ids[position + 1] for position in range(sequence_length))


def pair_from_text(
    text: str,
    vocabulary: CharacterVocabulary,
    sequence_length: int,
) -> tuple[list[int], list[int]] | None:
    """Encode real text and take one fixed window. Too-short text returns None."""
    return autoregressive_window(vocabulary.encode(text), sequence_length)


def build_batch(pairs: list[tuple[list[int], list[int]]]) -> dict[str, object]:
    """Stack pairs into a rectangular batch. Rows must already share one length."""
    if not pairs:
        raise ValueError("Cannot build an empty batch.")
    inputs = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    sequence_length = len(inputs[0])
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive.")
    for row in inputs + targets:
        if len(row) != sequence_length:
            raise ValueError("Every row in a batch must have the same length.")
    return {
        "input_ids": inputs,
        "target_ids": targets,
        "batch_size": len(pairs),
        "sequence_length": sequence_length,
    }


def ids_within_vocabulary(token_ids: list[int], vocabulary_size: int) -> bool:
    """Return whether every id is in ``[0, vocabulary_size)``."""
    return all(isinstance(token_id, int) and not isinstance(token_id, bool) and 0 <= token_id < vocabulary_size for token_id in token_ids)
