"""Training-only vocabulary for randomly initialized embeddings.

No pretrained vectors are loaded. Integer ids are the only representation
stored for Phase 6.
"""

from __future__ import annotations

from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


class Vocabulary:
    """Maps processed tokens to integer ids."""

    def __init__(self, token_to_id: dict[str, int], minimum_frequency: int, maximum_size: int, documents_fit: int):
        if token_to_id.get(PAD_TOKEN) != PAD_ID or token_to_id.get(UNK_TOKEN) != UNK_ID:
            raise ValueError("Reserved ids must place pad at 0 and unk at 1.")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = [""] * len(token_to_id)
        for token, index in token_to_id.items():
            self.id_to_token[index] = token
        self.minimum_frequency = minimum_frequency
        self.maximum_size = maximum_size
        self.documents_fit = documents_fit

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.encode_token(token) for token in tokens]

    def metadata(self) -> dict[str, object]:
        return {
            "size": len(self),
            "minimum_frequency": self.minimum_frequency,
            "maximum_size": self.maximum_size,
            "documents_fit": self.documents_fit,
            "reserved_tokens": [PAD_TOKEN, UNK_TOKEN],
            "padding_id": PAD_ID,
            "unknown_id": UNK_ID,
            "pretrained_vectors_loaded": False,
            "embedding_policy": "EMBEDDINGS WILL BE LEARNED FROM SCRATCH DURING MODEL TRAINING.",
        }


def build_vocabulary(token_counts: Counter[str], minimum_frequency: int, maximum_size: int, documents_fit: int) -> Vocabulary:
    """Build a deterministic vocabulary from training token counts.

    Tokens are ordered by descending count, then by token text. Reserved
    tokens occupy ids 0 and 1 and are not taken from the counts.
    """
    if minimum_frequency < 1 or maximum_size < 1:
        raise ValueError("Frequency and vocabulary size must be positive.")
    eligible = [(token, count) for token, count in token_counts.items() if count >= minimum_frequency and token not in (PAD_TOKEN, UNK_TOKEN)]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    kept = eligible[:maximum_size]
    token_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _count in kept:
        token_to_id[token] = len(token_to_id)
    return Vocabulary(token_to_id, minimum_frequency, maximum_size, documents_fit)
