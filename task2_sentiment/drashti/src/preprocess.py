"""Deterministic review preprocessing.

Stemming and lemmatization are not applied. A stemmer would merge distinct
surface forms before the later error review, and this phase does not download
a lemmatizer or any pretrained language model. The embedding layer in Phase 6
can learn related forms from the tokens kept here.
"""

from __future__ import annotations

import unicodedata

NEGATION_TOKENS = frozenset(
    {
        "not",
        "no",
        "never",
        "nor",
        "neither",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "without",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "mustnt",
        "shouldnt",
        "wouldnt",
        "couldnt",
        "but",
    }
)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "if",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
    }
)

_APOSTROPHES = frozenset("'`\u2018\u2019\u02bc")

if STOPWORDS & NEGATION_TOKENS:
    raise RuntimeError("Negation tokens must not also be stopwords.")


def preprocess_text(text: str) -> list[str]:
    """Return processed tokens for one review string.

    Steps: NFKC normalization, casefold, apostrophe removal, replacement of
    every other non-letter and non-number with whitespace, whitespace split,
    then stopword removal. Negation tokens are not stopwords. No stemmer runs.
    """
    if not isinstance(text, str):
        raise TypeError("preprocess_text expects a string.")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters: list[str] = []
    for character in normalized:
        if character in _APOSTROPHES:
            continue
        if character.isascii() and character.isalnum():
            characters.append(character)
        elif character.isalpha() or character.isdigit():
            characters.append(character)
        else:
            characters.append(" ")
    tokens = "".join(characters).split()
    return [token for token in tokens if token not in STOPWORDS]


def contains_negation(tokens: list[str]) -> bool:
    """Return whether any processed token is a preserved negation or contrast word."""
    return any(token in NEGATION_TOKENS for token in tokens)
