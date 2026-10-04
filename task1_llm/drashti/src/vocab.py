"""Character-level vocabulary built from story text.

No pretrained tokenizer, BPE, WordPiece, SentencePiece, or word tokenizer is used.
"""

from __future__ import annotations

from collections.abc import Iterable

CONSTRUCTION_POLICY = (
    "Collect every Unicode character that occurs in the 100,000 selected training stories. "
    "Sort those characters by code point and assign contiguous ids starting at 0. "
    "Do not lowercase, strip, or replace characters. "
    "Do not add special tokens. "
    "Do not fit the vocabulary on validation stories."
)
SPECIAL_TOKEN_POLICY = "No special tokens. Each training character is its own vocabulary item."


class CharacterVocabulary:
    """Maps characters to ids and ids back to characters."""

    def __init__(self, char_to_idx: dict[str, int]):
        self.char_to_idx = dict(char_to_idx)
        self.idx_to_char = {index: char for char, index in self.char_to_idx.items()}
        self._validate()

    def _validate(self) -> None:
        if not self.char_to_idx:
            raise ValueError("Vocabulary is empty.")
        if len(self.idx_to_char) != len(self.char_to_idx):
            raise ValueError("char_to_idx must be one-to-one.")
        if any(len(char) != 1 for char in self.char_to_idx):
            raise ValueError("Vocabulary keys must be single characters.")
        expected = list(range(len(self.idx_to_char)))
        if sorted(self.idx_to_char) != expected:
            raise ValueError("Vocabulary indices must be contiguous from 0.")

    @property
    def size(self) -> int:
        return len(self.char_to_idx)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        try:
            return [self.char_to_idx[char] for char in text]
        except KeyError as exc:
            raise ValueError("Text contains a character outside the vocabulary.") from exc

    def decode(self, token_ids: list[int]) -> str:
        if not isinstance(token_ids, list):
            raise TypeError("Token ids must be a list.")
        chars: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token ids must be integers.")
            try:
                chars.append(self.idx_to_char[token_id])
            except KeyError as exc:
                raise ValueError("Token id is outside the vocabulary.") from exc
        return "".join(chars)

    def contains_id(self, token_id: int) -> bool:
        return token_id in self.idx_to_char


def build_vocabulary(texts: Iterable[str]) -> CharacterVocabulary:
    """Build a vocabulary from the supplied texts and no other source."""
    found: set[str] = set()
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Vocabulary texts must be strings.")
        found.update(text)
    ordered = sorted(found)
    mapping = {char: index for index, char in enumerate(ordered)}
    return CharacterVocabulary(mapping)


def character_summary(vocabulary: CharacterVocabulary) -> dict[str, object]:
    """Describe the measured vocabulary without adding tokens."""
    non_ascii = [char for char in vocabulary.char_to_idx if ord(char) > 127]
    nonprintable = [char for char in vocabulary.char_to_idx if not char.isprintable()]
    return {
        "vocabulary_size": vocabulary.size,
        "construction_policy": CONSTRUCTION_POLICY,
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "non_ascii_count": len(non_ascii),
        "non_ascii_code_points": [ord(char) for char in non_ascii],
        "nonprintable_count": len(nonprintable),
        "nonprintable_code_points": [ord(char) for char in nonprintable],
    }


def vocabulary_payload(vocabulary: CharacterVocabulary) -> dict[str, object]:
    """JSON-ready vocabulary. Ids are stored as strings because JSON object keys are strings."""
    summary = character_summary(vocabulary)
    summary["char_to_idx"] = vocabulary.char_to_idx
    summary["idx_to_char"] = {str(index): char for index, char in vocabulary.idx_to_char.items()}
    return summary


def vocabulary_from_payload(payload: dict[str, object]) -> CharacterVocabulary:
    """Restore a vocabulary saved by ``vocabulary_payload``."""
    mapping = payload["char_to_idx"]
    if not isinstance(mapping, dict):
        raise TypeError("char_to_idx must be an object.")
    return CharacterVocabulary({str(char): int(index) for char, index in mapping.items()})
