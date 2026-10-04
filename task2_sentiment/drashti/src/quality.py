"""Inventory checks for Yelp Polarity rows.

Nothing is deleted. Empty and whitespace-only reviews are counted and kept
in the split inventory. A row is malformed only when the text is missing or
not a string, or the label is missing or outside the official class indices.
"""

from __future__ import annotations

import hashlib
from collections import Counter

ALLOWED_LABELS = (0, 1)


def _plain_label(label: object) -> object:
    """Convert an integer-like array scalar to a Python int. Booleans stay booleans."""
    if isinstance(label, bool) or label is None or isinstance(label, int):
        return label
    item = getattr(label, "item", None)
    if item is None:
        return label
    value = item()
    if isinstance(value, bool) or not isinstance(value, int):
        return label
    return value


def record_issues(text: object, label: object) -> list[str]:
    """Return every quality issue found on one row."""
    label = _plain_label(label)
    issues: list[str] = []
    if text is None:
        issues.append("missing_text")
    elif not isinstance(text, str):
        issues.append("non_string_text")
    elif text == "":
        issues.append("empty_text")
    elif text.strip() == "":
        issues.append("whitespace_only_text")
    if label is None:
        issues.append("missing_label")
    elif isinstance(label, bool) or not isinstance(label, int) or label not in ALLOWED_LABELS:
        issues.append("unexpected_label")
    return issues


def is_malformed(issues: list[str]) -> bool:
    """Malformed means the row cannot be trusted as text with an official label."""
    return any(issue in issues for issue in ("missing_text", "non_string_text", "missing_label", "unexpected_label"))


def empty_digest() -> dict[str, int]:
    """Start duplicate counters."""
    return {}


def text_digest(text: str) -> bytes:
    return hashlib.md5(text.encode("utf-8")).digest()


def audit_rows(rows: list[tuple[object, object]]) -> dict[str, object]:
    """Audit an in-memory fixture. Real splits use ``audit_dataset_split``."""
    return _audit(rows)


def _audit(rows: list[tuple[object, object]]) -> dict[str, object]:
    issue_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    malformed = 0
    character_lengths: list[int] = []
    word_lengths: list[int] = []
    text_counts: dict[bytes, int] = {}
    pair_counts: dict[bytes, int] = {}
    for text, label in rows:
        label = _plain_label(label)
        issues = record_issues(text, label)
        issue_counts.update(issues)
        if is_malformed(issues):
            malformed += 1
            continue
        assert isinstance(text, str)
        assert isinstance(label, int)
        class_counts[label] += 1
        character_lengths.append(len(text))
        word_lengths.append(len(text.split()))
        digest = text_digest(text)
        text_counts[digest] = text_counts.get(digest, 0) + 1
        pair = hashlib.md5(digest + str(label).encode("ascii")).digest()
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return _summarize_audit(len(rows), issue_counts, class_counts, malformed, character_lengths, word_lengths, text_counts, pair_counts)


def audit_dataset_split(dataset: object, text_field: str, label_field: str) -> dict[str, object]:
    """Audit one Hugging Face split without copying the review text."""
    issue_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    malformed = 0
    character_lengths: list[int] = []
    word_lengths: list[int] = []
    text_counts: dict[bytes, int] = {}
    pair_counts: dict[bytes, int] = {}
    total = len(dataset)
    seen = 0
    for batch in dataset.iter(batch_size=1000):
        for text, label in zip(batch[text_field], batch[label_field]):
            label = _plain_label(label)
            issues = record_issues(text, label)
            issue_counts.update(issues)
            if is_malformed(issues):
                malformed += 1
            else:
                assert isinstance(text, str)
                assert isinstance(label, int)
                class_counts[label] += 1
                character_lengths.append(len(text))
                word_lengths.append(len(text.split()))
                digest = text_digest(text)
                text_counts[digest] = text_counts.get(digest, 0) + 1
                pair = hashlib.md5(digest + str(label).encode("ascii")).digest()
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            seen += 1
            if seen % 50000 == 0:
                print(f"AUDIT {seen}/{total}", flush=True)
    return _summarize_audit(total, issue_counts, class_counts, malformed, character_lengths, word_lengths, text_counts, pair_counts)


def _duplicate_stats(counts: dict[bytes, int]) -> dict[str, int]:
    repeated_groups = 0
    records_in_repeated_groups = 0
    extra_copies = 0
    for count in counts.values():
        if count > 1:
            repeated_groups += 1
            records_in_repeated_groups += count
            extra_copies += count - 1
    return {
        "unique_keys": len(counts),
        "keys_with_more_than_one_record": repeated_groups,
        "records_in_repeated_groups": records_in_repeated_groups,
        "extra_copies": extra_copies,
    }


def _summarize_audit(
    total: int,
    issue_counts: Counter[str],
    class_counts: Counter[int],
    malformed: int,
    character_lengths: list[int],
    word_lengths: list[int],
    text_counts: dict[bytes, int],
    pair_counts: dict[bytes, int],
) -> dict[str, object]:
    counted = sum(class_counts.values())
    percentages = {str(label): (class_counts[label] / counted if counted else 0.0) for label in (0, 1)}
    majority = max(class_counts.values()) if class_counts else 0
    minority = min(class_counts.values()) if class_counts else 0
    return {
        "records": total,
        "issue_counts": dict(issue_counts),
        "malformed_records": malformed,
        "class_counts": {str(label): class_counts[label] for label in (0, 1)},
        "class_percentages": percentages,
        "class_imbalance_ratio_majority_over_minority": (majority / minority) if minority else None,
        "character_lengths": character_lengths,
        "word_lengths": word_lengths,
        "duplicate_text": _duplicate_stats(text_counts),
        "duplicate_text_and_label": _duplicate_stats(pair_counts),
        "rows_deleted": 0,
    }
